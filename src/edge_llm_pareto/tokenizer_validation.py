"""Live vLLM input/output tokenizer validation, outside measured trials."""
import json

import httpx

from .likelihood import ReferenceTokenizer, tokenize, detokenize


async def validate_vllm(*, base_url, model, reference, tasks, transport=None):
    """Check every question/answer and independently decode generated token IDs.

    The diagnostic generations have a small budget; this is not the
    experiment's output policy. It keeps a broken tokenizer check bounded.
    """
    mismatches = []
    texts = [(task['id'] + ':prompt', task['prompt']) for task in tasks]
    texts += [(task['id'] + ':answer', answer) for task in tasks for answer in task['answers']]
    texts += [('whitespace', 'Between September and November 1946\nTwo  spaces.\tTab.'),
              ('unicode', 'Copenhagen: København. café — 日本語')]
    async with httpx.AsyncClient(timeout=600, transport=transport) as client:
        for name, text in texts:
            ids = reference.encode(text)
            server_ids = await tokenize(client, 'vllm', base_url, model, text)
            if server_ids != ids:
                mismatches.append(dict(case=name, kind='encode'))
            decoded = await detokenize(client, 'vllm', base_url, model, ids)
            expected = reference.decode(ids)
            if decoded != expected:
                mismatches.append(dict(case=name, kind='decode', expected=expected, got=decoded))

        generations = []
        for chat, streaming in ((False, False), (False, True), (True, False), (True, True)):
            body = dict(model=model, temperature=0, max_tokens=32,
                        logprobs=True if chat else 1, return_tokens_as_token_ids=True, stream=streaming)
            if chat:
                body.update(messages=[dict(role='user', content='What is the capital of France?')], top_logprobs=0)
            else:
                body['prompt'] = 'The capital of France is'
            endpoint = base_url.rstrip('/') + ('/chat/completions' if chat else '/completions')
            tokens, parts, usage = [], [], None
            if streaming:
                body['stream_options'] = {'include_usage': True}
                chunks = []
                async with client.stream('POST', endpoint, json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data:') and line[5:].strip() != '[DONE]':
                            chunks.append(json.loads(line[5:]))
            else:
                response = await client.post(endpoint, json=body)
                response.raise_for_status()
                chunks = [response.json()]
            for chunk in chunks:
                usage = chunk.get('usage') or usage
                for choice in chunk.get('choices', []):
                    if chat:
                        message = choice.get('delta' if streaming else 'message') or {}
                        parts.append(message.get('content') or '')
                        tokens.extend(entry['token'] for entry in (choice.get('logprobs') or {}).get('content') or [])
                    else:
                        parts.append(choice.get('text') or '')
                        tokens.extend((choice.get('logprobs') or {}).get('tokens') or [])
            # These are actual sampled IDs, not a re-tokenization of the text.
            ids = [int(token.removeprefix('token_id:')) for token in tokens
                   if isinstance(token, str) and token.startswith('token_id:')]
            text = ''.join(parts)
            mode = ('chat_' if chat else 'completion_') + ('stream' if streaming else 'nonstream')
            exact = bool(ids) and len(ids) == len(tokens)
            if not exact:
                mismatches.append(dict(case=mode, kind='missing_output_token_ids'))
            elif reference.decode(ids) != text:
                mismatches.append(dict(case=mode, kind='generated_decode',
                                       expected=reference.decode(ids), got=text))
            if not usage or usage.get('completion_tokens') != len(ids):
                mismatches.append(dict(case=mode, kind='output_usage'))
            generations.append(dict(mode=mode, text=text, token_ids=ids, usage=usage))
    return dict(valid=not mismatches, reference=reference.identity(),
                input_cases=len(texts), generations=generations, mismatches=mismatches)


async def check_running_server(*, base_url, model, tokenizer_repo, tokenizer_revision, tasks):
    reference = ReferenceTokenizer(tokenizer_repo, tokenizer_revision)
    return await validate_vllm(base_url=base_url, model=model, reference=reference, tasks=tasks)

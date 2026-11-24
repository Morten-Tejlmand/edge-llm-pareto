"""The live checker must detect the pilot's input/output tokenizer corruption."""
import asyncio
import json
import unittest

import httpx

from edge_llm_pareto.tokenizer_validation import validate_vllm


class Reference:
    def encode(self, text):
        return [0] + [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(chr(i) for i in ids if i)

    def identity(self):
        return dict(repo='reference', revision='pinned')


def server(corruption=None):
    reference = Reference()

    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            text = body['prompt'].replace(' ', '') if corruption == 'encode' else body['prompt']
            return httpx.Response(200, json=dict(tokens=reference.encode(text)))
        if request.url.path == '/detokenize':
            text = reference.decode(body['tokens'])
            if corruption == 'decode':
                text = text.replace(' ', 'Ġ').replace('\n', 'Ċ')
            return httpx.Response(200, json=dict(prompt=text))
        if request.url.path in ('/v1/completions', '/v1/chat/completions'):
            text = 'Between September\nKøbenhavn.'
            ids = reference.encode(text)[1:] + [0]  # Include a special EOS token.
            if corruption == 'output':
                text = text.replace(' ', 'Ġ').replace('\n', 'Ċ')
            tokens = [f'token_id:{i}' for i in ids]
            if corruption == 'missing':
                tokens = []
            usage = dict(completion_tokens=len(ids) + (1 if corruption == 'usage' else 0))
            def chunk(text, tokens):
                if request.url.path == '/v1/chat/completions':
                    field = 'delta' if body['stream'] else 'message'
                    return dict(choices=[{field: dict(content=text),
                        'logprobs': dict(content=[dict(token=t) for t in tokens])}])
                return dict(choices=[dict(text=text, logprobs=dict(tokens=tokens))])
            if body['stream']:
                chunks = [chunk(text[:8], tokens[:8]), chunk(text[8:], tokens[8:]), dict(choices=[], usage=usage)]
                frames = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
                return httpx.Response(200, text=frames, headers={'content-type': 'text/event-stream'})
            return httpx.Response(200, json=dict(chunk(text, tokens), usage=usage))
        return httpx.Response(404)
    return httpx.MockTransport(handle)


class TokenizerValidationTests(unittest.TestCase):
    def check(self, corruption=None):
        tasks = [dict(id='one', prompt='Who sells lesson plans?', answers=['A teacher', 'Teachers'])]
        return asyncio.run(validate_vllm(base_url='http://test/v1', model='test',
            reference=Reference(), tasks=tasks, transport=server(corruption)))

    def test_correct_input_output_and_stream_usage_pass(self):
        record = self.check()
        self.assertTrue(record['valid'])
        self.assertEqual(record['input_cases'], 5)
        self.assertEqual(len(record['generations']), 4)

    def test_corruption_and_missing_usage_are_rejected(self):
        for corruption, expected in [('encode', 'encode'), ('decode', 'decode'),
                                     ('output', 'generated_decode'), ('usage', 'output_usage'),
                                     ('missing', 'missing_output_token_ids')]:
            with self.subTest(corruption=corruption):
                record = self.check(corruption)
                self.assertFalse(record['valid'])
                self.assertIn(expected, [m['kind'] for m in record['mismatches']])

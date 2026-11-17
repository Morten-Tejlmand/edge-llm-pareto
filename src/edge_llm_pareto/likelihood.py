"""Teacher-forced answer likelihood on the QA set: perplexity and KL divergence.

Each QA item is scored as the fixed text ``{prompt}\\n\\nAnswer: {reference answer}``.
The text is tokenized once per model and the same token IDs are sent to every
engine and artifact, so all of them score exactly the same tokens. For every
answer token the engine returns the log-probability of the reference token and
the top-k next-token log-probabilities.

- vLLM: one /v1/completions request per item with ``prompt_logprobs=k``.
- llama.cpp: llama-server has no prompt log-probabilities, so each answer position
  is one /completion request with ``n_predict=1`` and ``n_probs=k`` (raw softmax
  with ``temperature=-1``). If the reference token is outside the top k, the
  request is repeated with a larger ``n_probs``.

Perplexity is ``exp(mean answer-token NLL)``. KL(reference || candidate) per
position uses the reference top-k plus one bucket for the remaining mass; a
candidate token missing from its own top-k gets the candidate's smallest top-k
probability. This is an approximation of the full-vocabulary KL, recorded as
``KL_METHOD``. Raw text, not the chat template, is scored on purpose: it is a
likelihood probe of the weights, identical across engines.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEMA_VERSION = 1
TEXT_FORMAT = 'prompt_blank_line_answer_colon_v1'
KL_METHOD = 'topk_reference_plus_tail_bucket_v1'
DEFAULT_TOP_K = 20  # vLLM's default --max-logprobs
LLAMA_FALLBACK_PROBS = (1000, 200000)


def item_texts(task):
    context = f"{task['prompt'].rstrip()}\n\nAnswer:"
    return context, context + ' ' + task['answers'][0].strip()


def load_tasks(path):
    raw = Path(path).read_bytes()
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('QA dataset requires unique ids')
    for t in tasks:
        if not isinstance(t.get('prompt'), str) or not t.get('answers'):
            raise ValueError('QA dataset requires prompt and answers per item')
    return tasks, hashlib.sha256(raw).hexdigest()


def split_answer(context_ids, full_ids):
    """Answer token IDs, or None when tokenization is not prefix-stable."""
    n = len(context_ids)
    if full_ids[:n] != context_ids or len(full_ids) == n:
        return None
    return full_ids[n:]


def sequences_sha(items):
    payload = json.dumps([[i['id'], i['context_ids'], i['answer_ids']] for i in items])
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------------------ server I/O

def root_url(base_url):
    return base_url.rsplit('/v1', 1)[0]


async def tokenize(client, engine, base_url, model, text):
    if engine == 'llama_cpp':
        r = await client.post(root_url(base_url) + '/tokenize',
                              json={'content': text, 'add_special': True})
        r.raise_for_status()
        return [t if isinstance(t, int) else t['id'] for t in r.json()['tokens']]
    r = await client.post(root_url(base_url) + '/tokenize',
                          json={'model': model, 'prompt': text, 'add_special_tokens': True})
    r.raise_for_status()
    return r.json()['tokens']


async def build_sequences(client, engine, base_url, model, tasks):
    items, skipped = [], []
    for task in tasks:
        context, full = item_texts(task)
        context_ids = await tokenize(client, engine, base_url, model, context)
        full_ids = await tokenize(client, engine, base_url, model, full)
        answer_ids = split_answer(context_ids, full_ids)
        if answer_ids is None:
            skipped.append(task['id'])
            continue
        items.append(dict(id=task['id'], context_ids=context_ids, answer_ids=answer_ids))
    return items, skipped


def parse_vllm_prompt_logprobs(prompt_logprobs, ids, start, k):
    """Positions start..len(ids)-1 from a vLLM prompt_logprobs list."""
    positions = []
    for i in range(start, len(ids)):
        entry = {int(t): v for t, v in (prompt_logprobs[i] or {}).items()}
        target = entry.get(ids[i])
        if target is None:
            raise ValueError(f'vLLM omitted the prompt token at position {i}')
        top = sorted(((t, v['logprob']) for t, v in entry.items()
                      if v.get('rank') is not None and v['rank'] <= k), key=lambda x: -x[1])
        positions.append(dict(target=ids[i], target_logprob=target['logprob'],
                              top=[[t, lp] for t, lp in top[:k]]))
    return positions


def parse_llama_probs(body, target, k):
    """(position record or None if target missing, smallest top logprob)."""
    probs = body['completion_probabilities'][0]
    top = sorted(((p['id'], p['logprob']) for p in probs['top_logprobs']), key=lambda x: -x[1])
    lookup = dict(top)
    if target not in lookup:
        return None
    return dict(target=target, target_logprob=lookup[target], top=[[t, lp] for t, lp in top[:k]])


async def score_vllm(client, base_url, model, item, k):
    ids = item['context_ids'] + item['answer_ids']
    r = await client.post(base_url + '/completions', json={
        'model': model, 'prompt': ids, 'max_tokens': 1, 'temperature': 0,
        'prompt_logprobs': k})
    r.raise_for_status()
    return parse_vllm_prompt_logprobs(r.json()['choices'][0]['prompt_logprobs'],
                                      ids, len(item['context_ids']), k)


async def score_llama(client, base_url, model, item, k):
    ids = item['context_ids'] + item['answer_ids']
    positions = []
    for i in range(len(item['context_ids']), len(ids)):
        record = None
        for n_probs in (k, *LLAMA_FALLBACK_PROBS):
            r = await client.post(root_url(base_url) + '/completion', json={
                'prompt': ids[:i], 'n_predict': 1, 'n_probs': n_probs, 'temperature': -1,
                'post_sampling_probs': False, 'cache_prompt': False})
            r.raise_for_status()
            record = parse_llama_probs(r.json(), ids[i], k)
            if record is not None:
                break
        if record is None:
            raise ValueError(f'reference token {ids[i]} not returned by llama-server')
        positions.append(record)
    return positions


async def score_items(client, engine, base_url, model, items, k, concurrency=8):
    scorer = score_llama if engine == 'llama_cpp' else score_vllm
    gate = asyncio.Semaphore(concurrency)

    async def one(item):
        async with gate:
            return dict(id=item['id'], positions=await scorer(client, base_url, model, item, k))
    return await asyncio.gather(*(one(i) for i in items))


def load_or_build_tokens(cache_path, built, *, base_model, dataset_sha256, engine):
    """Reuse the model's frozen token IDs; the first engine to run creates them."""
    cache = Path(cache_path)
    if cache.exists():
        frozen = json.loads(cache.read_text())
        if frozen['base_model'] != base_model or frozen['dataset_sha256'] != dataset_sha256:
            raise ValueError(f'{cache} belongs to another model or dataset')
        return frozen
    items, skipped = built
    frozen = dict(schema_version=SCHEMA_VERSION, base_model=base_model,
                  dataset_sha256=dataset_sha256, text_format=TEXT_FORMAT,
                  tokenized_by_engine=engine, skipped_ids=skipped, items=items,
                  sequences_sha256=sequences_sha(items))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(frozen))
    return frozen


async def evaluate(*, base_url, engine, model, base_model, label, model_repo, model_revision,
                   dataset, tokens_dir, output, top_k=DEFAULT_TOP_K, concurrency=8, timeout=600,
                   transport=None):
    """Score one running server and write its likelihood record."""
    import httpx
    tasks, dataset_sha = load_tasks(dataset)
    cache = Path(tokens_dir) / f'{base_model}-{dataset_sha[:12]}.json'
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        built = None if cache.exists() else await build_sequences(client, engine, base_url, model, tasks)
        frozen = load_or_build_tokens(cache, built, base_model=base_model,
                                      dataset_sha256=dataset_sha, engine=engine)
        # Token IDs are sent directly, so the server tokenizer is only checked, not used.
        probe = frozen['items'][0]
        context = item_texts(next(t for t in tasks if t['id'] == probe['id']))[0]
        tokenizer_matches = await tokenize(client, engine, base_url, model, context) == probe['context_ids']
        scored = await score_items(client, engine, base_url, model, frozen['items'], top_k, concurrency)
    record = dict(schema_version=SCHEMA_VERSION, base_model=base_model, engine=engine, label=label,
                  model_repo=model_repo, model_revision=model_revision,
                  dataset_sha256=dataset_sha, text_format=TEXT_FORMAT, top_k=top_k,
                  sequences_sha256=frozen['sequences_sha256'], tokens_file=str(cache),
                  tokenizer_matches_frozen_ids=tokenizer_matches,
                  summary=summarize(scored), items=scored)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record))
    return record


# ------------------------------------------------------------------- metrics

def summarize(items):
    nll = [-p['target_logprob'] for i in items for p in i['positions']]
    if not nll:
        raise ValueError('No scored answer tokens')
    mean = statistics.fmean(nll)
    return dict(items=len(items), answer_tokens=len(nll), answer_nll=mean,
                answer_ppl=math.exp(mean))


def _logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def position_kl(reference, candidate):
    """KL(reference || candidate) over reference top-k plus a tail bucket."""
    ref = dict((t, lp) for t, lp in reference['top'])
    cand = dict((t, lp) for t, lp in candidate['top'])
    cand[candidate['target']] = candidate['target_logprob']
    floor = min(lp for _, lp in candidate['top'])
    kl, p_mass, q_mass = 0.0, 0.0, 0.0
    for t, lp in ref.items():
        lq = cand.get(t, floor)
        p = math.exp(lp)
        kl += p * (lp - lq)
        p_mass += p
        q_mass += math.exp(lq)
    p_tail, q_tail = max(1 - p_mass, 1e-12), max(1 - q_mass, 1e-12)
    return max(0.0, kl + p_tail * math.log(p_tail / q_tail))


def _interval(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def compare(reference, candidate):
    """KL, top-1 agreement and NLL change of candidate against reference."""
    if reference['sequences_sha256'] != candidate['sequences_sha256']:
        raise ValueError('Likelihood records scored different token sequences')
    ref_items = {i['id']: i for i in reference['items']}
    per_item_kl, agree, total = [], 0, 0
    for item in candidate['items']:
        ref_item = ref_items[item['id']]
        kls = []
        for rp, cp in zip(ref_item['positions'], item['positions'], strict=True):
            kls.append(position_kl(rp, cp))
            agree += rp['top'][0][0] == cp['top'][0][0]
            total += 1
        per_item_kl.append(statistics.fmean(kls))
    kl, low, high = _interval(per_item_kl)
    return dict(kl_mean=kl, kl_ci_low=low, kl_ci_high=high, top1_agreement=agree / total,
                answer_nll_delta=candidate['summary']['answer_nll'] - reference['summary']['answer_nll'],
                kl_method=KL_METHOD, reference=f"{reference['engine']}/{reference['label']}")

#!/usr/bin/env python3
"""Score fixed JSONL answers with exact match or corpus BLEU."""
import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path


def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().casefold())


def corpus_bleu(predictions, references):
    """Multi-reference BLEU on a 0–100 scale; missing references are ignored."""
    from sacrebleu.metrics import BLEU
    if not predictions or len(predictions) != len(references) or any(not r for r in references):
        raise ValueError('BLEU requires aligned predictions and nonempty reference lists')
    streams = [[refs[i] if i < len(refs) else None for refs in references]
               for i in range(max(map(len, references)))]
    metric = BLEU(tokenize='13a', lowercase=False, smooth_method='exp', effective_order=True)
    score = metric.corpus_score(predictions, streams)
    return dict(score=min(100.0, max(0.0, score.score)), signature=str(metric.get_signature()),
                brevity_penalty=score.bp, precisions=score.precisions,
                system_length=score.sys_len, reference_length=score.ref_len)


ANSWER_EXTRACTION = 'think_blocks_v2'


def extract_answer(raw, finish_reason=None):
    """Remove leading reasoning blocks and return (answer, status).

    v2 handles two cases v1 (``leading_think_blocks_v1``) scored as wrong answers:
    - Prompt-opened reasoning: some chat templates (Qwen3.8 on vLLM) end the
      generation prompt with ``<think>``, so the completion holds only the
      closing tag. Text before a first ``</think>`` that has no opening tag is
      treated as reasoning.
    - Cut-off reasoning: a completion stopped by the output limit
      (``finish_reason == 'length'``) before any ``</think>`` is incomplete
      reasoning, whether or not the opening tag was generated.
    """
    answer = raw.strip()
    head, closed, _ = answer.partition('</think>')
    if closed and '<think>' not in head:
        answer = '<think>' + answer
    while answer.startswith('<think>'):
        end = answer.find('</think>', len('<think>'))
        if end < 0:
            return '', 'incomplete_reasoning'
        answer = answer[end + len('</think>'):].strip()
    if finish_reason == 'length' and not closed:
        return '', 'incomplete_reasoning'
    return answer, 'ok'


def score_results(record, tasks, scorer):
    results = record['results']
    by_id = {r['id']: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {t['id'] for t in tasks}:
        raise ValueError('Result IDs must match dataset IDs exactly')
    ordered = []
    for task in tasks:
        result = by_id[task['id']]
        raw = result.get('raw_answer', result.get('answer', ''))
        answer, status = extract_answer(raw, result.get('finish_reason'))
        result.update(raw_answer=raw, answer=answer, extraction_status=status,
                      correct=not result.get('error') and normalize(answer) in
                      [normalize(a) for a in task['answers']])
        ordered.append(result)
    exact_match = sum(r['correct'] for r in ordered) / len(ordered)
    bleu = corpus_bleu([r['answer'] if not r.get('error') else '' for r in ordered],
                       [t['answers'] for t in tasks])
    identity = ('sacrebleu_corpus_v1|' + bleu['signature'] if scorer == 'bleu'
                else 'normalized_exact_match_v2')
    record.update(schema_version=2, results=ordered, answer_extraction=ANSWER_EXTRACTION,
                  scorer=identity + '|' + ANSWER_EXTRACTION,
                  score=bleu['score'] / 100 if scorer == 'bleu' else exact_match,
                  metrics=dict(exact_match=exact_match, bleu=bleu),
                  failed=sum('error' in r for r in ordered),
                  incomplete_reasoning=sum(r['extraction_status'] == 'incomplete_reasoning' for r in ordered))
    return record


def score_measured_requests(results, workload):
    """Score measured responses after timing/power collection has ended.

    Request IDs identify occurrences; question IDs may repeat as the corpus
    cycles at higher concurrency. Failures remain in the scoring denominator.
    """
    tasks, predictions = [], []
    for result in results:
        task = workload['prompts'][result['request_id'] % len(workload['prompts'])]
        occurrence = str(result['request_id'])
        tasks.append(dict(id=occurrence, answers=task['answers']))
        prediction = dict(id=occurrence, raw_answer=result.get('raw_answer', ''),
                          finish_reason=result.get('finish_reason'))
        if not result['ok']:
            prediction['error'] = result.get('error') or 'request_failed'
        predictions.append(prediction)
        result.update(question_id=task['id'], reference_answers=task['answers'])
    scored = score_results(dict(results=predictions), tasks, workload['qa_scorer'])
    by_id = {r['id']: r for r in scored['results']}
    for result in results:
        prediction = by_id[str(result['request_id'])]
        result.update({k: prediction[k] for k in
                       ('raw_answer', 'answer', 'extraction_status', 'correct')})
    summary = {k: v for k, v in scored.items() if k != 'results'}
    summary.update(evaluation_mode='measured_qa', scoring_protocol='measured_requests_v1',
                   dataset_sha256=workload['qa_dataset_sha256'],
                   max_tokens=workload['max_output_tokens'],
                   output_limit_policy=workload['output_limit_policy'],
                   temperature=workload['temperature'], requests=len(results),
                   unique_questions=len({r['question_id'] for r in results}),
                   dataset_questions=len(workload['prompts']),
                   length_limited_requests=sum(r.get('finish_reason') == 'length' for r in results))
    return summary


def rescore(dataset, source, output):
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError('Use a separate output file to preserve the original evaluation')
    raw = Path(dataset).read_bytes()
    record = json.loads(Path(source).read_text())
    if record.get('dataset_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Dataset checksum does not match the original evaluation')
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    scorer = 'bleu' if record['scorer'].startswith('sacrebleu_corpus_v1|') else 'exact-match'
    record['rescored_from'] = dict(path=str(source), sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    score_results(record, tasks, scorer)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=2))
    print(f"Exact match: {record['metrics']['exact_match']:.3f}; corpus BLEU: {record['metrics']['bleu']['score']:.2f}/100; failures: {record['failed']}")
    return 1 if record['failed'] else 0


async def evaluate(args):
    from openai import AsyncOpenAI
    raw = Path(args.dataset).read_bytes()
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Dataset must contain unique IDs and at least one task')
    for task in tasks:
        if not isinstance(task.get('prompt'), str) or not isinstance(task.get('answers'), list) or not task['answers'] or not all(isinstance(a, str) and a.strip() for a in task['answers']):
            raise ValueError('Each task requires prompt and nonempty list of accepted answers')
    limit = getattr(args, 'max_tokens', None)
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError('max-tokens must be a positive integer')
    options = {'max_tokens': limit} if limit is not None else {}
    results = []
    async with AsyncOpenAI(base_url=args.base_url, api_key='EMPTY', max_retries=0, timeout=600) as client:
        for task in tasks:
            try:
                response = await client.chat.completions.create(model=args.model,
                    messages=[{'role': 'user', 'content': task['prompt']}],
                    temperature=0, **options)
                answer = response.choices[0].message.content or ''
                results.append(dict(id=task['id'], answer=answer,
                    finish_reason=getattr(response.choices[0], 'finish_reason', None),
                    correct=normalize(answer) in [normalize(a) for a in task['answers']]))
            except Exception as exc:
                results.append(dict(id=task['id'], answer='', correct=False, error=str(exc)))
    record = dict(schema_version=2, model=args.model, model_repo=args.model_repo,
                  model_revision=args.model_revision, base_model=args.base_model,
                  engine=args.engine, label=args.rung,
                  dataset_sha256=hashlib.sha256(raw).hexdigest(),
                  max_tokens=limit, output_limit_policy="explicit_cap_or_eos_or_context" if limit is not None else "eos_or_context",
                  temperature=0, results=results)
    score_results(record, tasks, getattr(args, 'scorer', 'exact-match'))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(f"Exact match: {record['metrics']['exact_match']:.3f}; corpus BLEU: {record['metrics']['bleu']['score']:.2f}/100; failures: {record['failed']}")
    return 1 if record['failed'] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    for flag in ('dataset', 'model', 'base-model', 'model-repo', 'model-revision', 'rung', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--engine', choices=['llama_cpp'], default='llama_cpp')
    p.add_argument('--max-tokens', type=int, default=None, help='QA output cap; default uncapped within context')
    p.add_argument('--scorer', choices=['exact-match', 'bleu'], default='exact-match',
                   help='primary quality score; both metrics are saved, BLEU is divided by 100 for filtering')
    args = p.parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    return asyncio.run(evaluate(args))


if __name__ == '__main__':
    raise SystemExit(main())

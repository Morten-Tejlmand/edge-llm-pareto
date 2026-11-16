"""Fixed prompt corpora, independent of the model's tokenizer."""
import hashlib
import json
from pathlib import Path


def attach_prompts(workload, path):
    raw = Path(path).read_bytes()
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not tasks or any(not isinstance(t, dict) or not isinstance(t.get('id'), str)
                        or not isinstance(t.get('prompt'), str) or not t['prompt'].strip() for t in tasks):
        raise ValueError('Prompt dataset requires nonempty string id/prompt records')
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Prompt IDs must be unique')
    return dict(workload, prompt_tokens=None, prompts=tasks,
                prompt_dataset_sha256=hashlib.sha256(raw).hexdigest())


def attach_qa(workload, path, scorer='exact-match'):
    """Attach reference answers for scoring the actual concurrent requests."""
    if scorer not in ('exact-match', 'bleu'):
        raise ValueError('Unknown QA scorer')
    result = attach_prompts(workload, path)
    for task in result['prompts']:
        answers = task.get('answers')
        if not isinstance(answers, list) or not answers or not all(
                isinstance(a, str) and a.strip() for a in answers):
            raise ValueError('QA dataset requires a nonempty list of accepted answers per task')
    return dict(result, evaluation_mode='measured_qa', qa_scorer=scorer,
                qa_dataset_sha256=result['prompt_dataset_sha256'])


def excerpts(text, words=192, count=100):
    if words < 1 or count < 1:
        raise ValueError('words and count must be positive')
    tokens = text.split()
    if len(tokens) < words * count:
        raise ValueError('Corpus too short for requested non-overlapping excerpts')
    return [dict(id=f'excerpt-{i:05d}', prompt='Continue the following passage:\n\n' +
                 ' '.join(tokens[i*words:(i+1)*words])) for i in range(count)]

# QA pilot dataset

`squad-dev-100.jsonl` is derived from the SQuAD 1.1 **development** split,
by Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang (2016).

- Project and attribution: https://rajpurkar.github.io/SQuAD-explorer/
- Paper: https://arxiv.org/abs/1606.05250
- Source: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
- Dataset license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  The derived JSONL subset is distributed under that same license.

Changes: select 100 questions by SHA-256 ranking of `42:<question-id>`, retain
original question IDs and article titles, combine passages and questions into a
fixed answer-only prompt, and deduplicate identical reference strings. No answer
text is added to the prompt. The metadata sidecar records the source and output
SHA-256 hashes, preparation version, seed, selected IDs and source count.

To reproduce or prepare a larger subset:

```sh
curl -fL https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json \
  -o /tmp/squad-dev-v1.1.json
.venv/bin/python scripts/prepare_qa.py --source /tmp/squad-dev-v1.1.json \
  --output data/squad-dev-100.jsonl --count 100 --seed 42
```

This subset is for pipeline testing and a small reading-comprehension pilot.
Use the same frozen sample across model variants. Sampling questions does not
make passages independent, and a public benchmark may have appeared in training
data. Do not treat 100 questions as sufficient evidence of broad model quality.
BLEU is a supplementary overlap metric; official SQuAD evaluation uses EM/F1.

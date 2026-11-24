import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rescore_trials', ROOT / 'scripts' / 'rescore_trials.py')
rescore_trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore_trials)


def trial(raw, finish='stop'):
    workload = dict(evaluation_mode='measured_qa', qa_scorer='exact-match', qa_dataset_sha256='s',
                    max_output_tokens=2048, output_limit_policy='explicit_cap_or_eos_or_context',
                    temperature=0.0, prompts=[dict(id='q1', prompt='P', answers=['Diepoldsau'])])
    return dict(scenario='generation_load', workload=workload, qa_exact_match=0.0, output_tok_s=10.0,
                quality=dict(metrics=dict(exact_match=0.0)),
                request_results=[dict(request_id=0, ok=True, raw_answer=raw, finish_reason=finish)])


class RescoreTrialsTest(unittest.TestCase):
    def test_prompt_opened_reasoning_is_rescored_without_touching_speed(self):
        row, changed = rescore_trials.rescore_row(trial('Need span.\n</think>\n\nDiepoldsau'))
        self.assertTrue(changed)
        self.assertEqual(row['qa_exact_match'], 1.0)
        self.assertEqual(row['quality']['answer_extraction'], 'think_blocks_v2')
        self.assertEqual(row['output_tok_s'], 10.0)

    def test_skipped_rows_pass_through_and_inputs_are_kept(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'in.jsonl'
            src.write_text(json.dumps(dict(skipped=True, label='x')) + '\n' + json.dumps(trial('a</think>Diepoldsau')) + '\n')
            original = src.read_text()
            import sys
            argv, sys.argv = sys.argv, ['rescore_trials.py', str(src), '--output-dir', str(Path(d) / 'out')]
            try:
                self.assertEqual(rescore_trials.main(), 0)
            finally:
                sys.argv = argv
            self.assertEqual(src.read_text(), original)
            rows = [json.loads(l) for l in (Path(d) / 'out' / 'in.jsonl').read_text().splitlines()]
            self.assertTrue(rows[0]['skipped'])
            self.assertEqual(rows[1]['qa_exact_match'], 1.0)
            self.assertIn('rescored_from', rows[1])


if __name__ == '__main__':
    unittest.main()

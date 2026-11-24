"""Study matrix construction: GGUF matrix plus native vLLM rows, no inference."""
import copy
import importlib.util
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qa_study', ROOT / 'scripts/qa_study.py')
qa_study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_study)
CONFIG_PATH = ROOT / 'configs/study.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())


NATIVE = {'qwen3.8-27b': ['native-bf16', 'qwen-fp8', 'nvidia-nvfp4'],
          'deepseek-r1-0528-qwen3-8b': ['native-bf16']}
OFFICIAL_PUBLISHERS = ('Qwen/', 'deepseek-ai/', 'nvidia/')


def build(config):
    return qa_study.build_commands(config, config_path=CONFIG_PATH, qa_dataset='qa.jsonl')


def rows(commands):
    return [(c[c.index('--model') + 1], c[c.index('--engine') + 1], c[c.index('--rung') + 1])
            for c in commands]


class StudyMatrixTest(unittest.TestCase):
    def test_full_dataset_study_defaults_and_pilot_override(self):
        commands, _, _ = build(CONFIG)
        for command in commands:
            self.assertNotIn('--duration-seconds', command)
            self.assertIn('--full-dataset', command)
            self.assertEqual(command[command.index('--output-tokens') + 1], 'none')
            if command[command.index('--engine') + 1] == 'vllm':
                self.assertIn('--validate-output-tokenizer', command)
        commands, _, _ = qa_study.build_commands(CONFIG, config_path=CONFIG_PATH,
            qa_dataset='qa.jsonl', requests=1)
        self.assertTrue(all('--duration-seconds' not in c for c in commands))
        self.assertTrue(all('--full-dataset' not in c for c in commands))
        commands, _, _ = qa_study.build_commands(CONFIG, config_path=CONFIG_PATH,
            qa_dataset='qa.jsonl', requests=1, duration_seconds=120)
        self.assertTrue(all(c[c.index('--duration-seconds') + 1] == '120' for c in commands))

    def test_main_study_is_1500_trials(self):
        commands, configurations, trials = build(CONFIG)
        # llama.cpp: 2 models x 3 GGUF rungs; vLLM: 3 Qwen + 1 DeepSeek native rungs;
        # each x 5 concurrency levels.
        self.assertEqual(configurations, (6 + 4) * 5)
        self.assertEqual(trials, 1500)
        self.assertEqual(sorted(rows(commands)), sorted([
            ('qwen3.8-27b', 'llama_cpp', 'gguf-bf16,gguf-ud-q8_k_xl,gguf-ud-q4_k_xl'),
            ('deepseek-r1-0528-qwen3-8b', 'llama_cpp', 'gguf-bf16,gguf-ud-q8_k_xl,gguf-ud-q4_k_xl'),
            ('qwen3.8-27b', 'vllm', 'native-bf16,qwen-fp8,nvidia-nvfp4'),
            ('deepseek-r1-0528-qwen3-8b', 'vllm', 'native-bf16')]))

    def test_every_sweep_scores_answer_likelihood(self):
        commands, _, _ = build(CONFIG)
        for c in commands:
            self.assertEqual(c[c.index('--likelihood-dataset') + 1], 'qa.jsonl')
        config = copy.deepcopy(CONFIG)
        config['study']['likelihood'] = False
        self.assertTrue(all('--likelihood-dataset' not in c for c in build(config)[0]))

    def test_gguf_engines_default_to_all_engines(self):
        config = copy.deepcopy(CONFIG)
        del config['study']['gguf_engines']
        config['study'].pop('native_vllm_rungs')
        _, configurations, _ = build(config)
        self.assertEqual(configurations, 2 * 3 * 2 * 5)

    def test_rejects_gguf_engine_outside_study_engines(self):
        config = copy.deepcopy(CONFIG)
        config['study']['gguf_engines'] = ['llama_cpp', 'sglang']
        with self.assertRaises(ValueError):
            build(config)

    def test_engine_subsets(self):
        llama = qa_study.subset(CONFIG, engines=['llama_cpp'])
        self.assertEqual({r[1] for r in rows(build(llama)[0])}, {'llama_cpp'})
        vllm = qa_study.subset(CONFIG, engines=['vllm'])
        self.assertEqual({r[1] for r in rows(build(vllm)[0])}, {'vllm'})
        self.assertEqual(build(vllm)[1], 4 * 5)
        with self.assertRaises(ValueError):  # GGUF rungs are not run on vLLM
            qa_study.subset(CONFIG, engines=['vllm'], rungs=['gguf-bf16'])

    def test_pilot_subset_filters_rungs_and_concurrency(self):
        config = qa_study.subset(CONFIG, rungs=['gguf-bf16', 'gguf-ud-q4_k_xl'],
                                 concurrency=[1, 16], engines=['llama_cpp'])
        commands, configurations, _ = build(config)
        self.assertEqual(configurations, 2 * 2 * 1 * 2)
        self.assertEqual({r[2] for r in rows(commands)}, {'gguf-bf16,gguf-ud-q4_k_xl'})
        self.assertTrue(all(c[c.index('--concurrency') + 1] == '1,16' for c in commands))
        self.assertEqual(CONFIG['concurrency'], [1, 2, 4, 8, 16])  # original untouched

    def test_pilot_subset_rejects_unknown_values(self):
        for kwargs in (dict(models=['nope']), dict(rungs=['gguf-q2']), dict(concurrency=[3])):
            with self.assertRaises(ValueError):
                qa_study.subset(CONFIG, **kwargs)

    def test_resume_attaches_matching_earlier_results(self):
        import json, tempfile
        commands, _, _ = build(CONFIG)
        with tempfile.TemporaryDirectory() as d:
            done = Path(d) / 'qwen3.8-27b-llama_cpp-20260101-000000.jsonl'
            done.write_text(json.dumps({'label': 'gguf-bf16', 'trial': 0}) + '\n')
            other = Path(d) / 'qwen3.8-27b-vllm-20260101-000000.jsonl'
            other.write_text(json.dumps({'label': 'unsloth-nvfp4'}) + '\n')  # not a study rung
            resumed = qa_study.add_resume(commands, d)
        self.assertEqual(resumed, [done])
        resumed_cmds = [c for c in commands if '--resume' in c]
        self.assertEqual(len(resumed_cmds), 1)
        self.assertEqual(resumed_cmds[0][resumed_cmds[0].index('--model') + 1], 'qwen3.8-27b')
        self.assertEqual(resumed_cmds[0][resumed_cmds[0].index('--engine') + 1], 'llama_cpp')

    def test_resume_refuses_ambiguous_earlier_results(self):
        import json, tempfile
        commands, _, _ = build(CONFIG)
        with tempfile.TemporaryDirectory() as d:
            for stamp in ('20260101-000000', '20260102-000000'):
                (Path(d) / f'qwen3.8-27b-llama_cpp-{stamp}.jsonl').write_text(
                    json.dumps({'label': 'gguf-bf16'}) + '\n')
            with self.assertRaises(ValueError):
                qa_study.add_resume(commands, d)

    def test_artifacts_are_pinned_unsloth_gguf_or_official_native(self):
        study = CONFIG['study']
        self.assertEqual(study['native_vllm_rungs'], NATIVE)
        cells = [(m, e, labels) for m, labels in study['model_rungs'].items()
                 for e in study['gguf_engines']]
        cells += [(m, 'vllm', labels) for m, labels in study['native_vllm_rungs'].items()]
        for model, engine, labels in cells:
            engine_cfg = CONFIG['models'][model]['engine_rungs'][engine]
            rungs = {r['rung']: r for r in engine_cfg['rungs']}
            for label in labels:
                rung = rungs[label]
                repo = rung.get('repo', engine_cfg['repo'])
                revision = rung.get('revision') if 'repo' in rung else engine_cfg.get('revision')
                self.assertRegex(revision or '', r'^[0-9a-f]{40}$', (model, label))
                if engine == 'llama_cpp':
                    self.assertTrue(repo.startswith('unsloth/'), (model, label, repo))
                else:
                    self.assertNotIn('file', rung, (model, label))
                    self.assertTrue(repo.startswith(OFFICIAL_PUBLISHERS), (model, label, repo))

    def test_rejects_native_rung_missing_from_vllm(self):
        config = copy.deepcopy(CONFIG)
        config['study']['native_vllm_rungs'] = {'deepseek-r1-0528-qwen3-8b': ['unsloth-nvfp4']}
        with self.assertRaises(ValueError):
            build(config)

    def test_rejects_native_rows_duplicating_gguf_matrix(self):
        config = copy.deepcopy(CONFIG)
        config['study']['native_vllm_rungs'] = {'qwen3.8-27b': ['gguf-ud-q8_k_xl']}
        with self.assertRaises(ValueError):
            build(config)

    def test_rejects_native_rows_without_vllm_engine(self):
        config = copy.deepcopy(CONFIG)
        config['study']['engines'] = ['llama_cpp']
        config['study']['gguf_engines'] = ['llama_cpp']
        with self.assertRaises(ValueError):
            build(config)


if __name__ == '__main__':
    unittest.main()

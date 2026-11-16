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


NATIVE = {'qwen3.8-27b': ['native-bf16', 'unsloth-nvfp4'],
          'deepseek-r1-0528-qwen3-8b': ['native-bf16']}


def with_native(config):
    config = copy.deepcopy(config)
    config['study']['native_vllm_rungs'] = copy.deepcopy(NATIVE)
    return config


def build(config):
    return qa_study.build_commands(config, config_path=CONFIG_PATH, qa_dataset='qa.jsonl')


def rows(commands):
    return [(c[c.index('--model') + 1], c[c.index('--engine') + 1], c[c.index('--rung') + 1])
            for c in commands]


class StudyMatrixTest(unittest.TestCase):
    def test_main_study_is_1800_trials(self):
        commands, configurations, trials = build(CONFIG)
        # 2 models x 3 GGUF rungs x 2 engines x 5 concurrency levels.
        self.assertEqual(configurations, 60)
        self.assertEqual(trials, 1800)
        self.assertEqual(len(commands), 4)

    def test_optional_native_vllm_rows(self):
        config = with_native(CONFIG)
        commands, configurations, trials = build(config)
        self.assertIn(('qwen3.8-27b', 'vllm', 'native-bf16,unsloth-nvfp4'), rows(commands))
        self.assertIn(('deepseek-r1-0528-qwen3-8b', 'vllm', 'native-bf16'), rows(commands))
        self.assertEqual(configurations, 60 + 15)
        self.assertEqual(trials, 75 * 30)

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

    def test_every_quantized_artifact_is_unsloth_and_pinned(self):
        for model, labels in CONFIG['study']['model_rungs'].items():
            for engine in CONFIG['study']['engines']:
                engine_cfg = CONFIG['models'][model]['engine_rungs'][engine]
                rungs = {r['rung']: r for r in engine_cfg['rungs']}
                for label in labels + NATIVE.get(model, []):
                    if label not in rungs:
                        continue
                    rung = rungs[label]
                    repo = rung.get('repo', engine_cfg['repo'])
                    revision = rung.get('revision') if 'repo' in rung else engine_cfg.get('revision')
                    self.assertRegex(revision or '', r'^[0-9a-f]{40}$', (model, label))
                    if rung['linear_bits'] < 16:
                        self.assertTrue(repo.startswith('unsloth/'), (model, label, repo))

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
        config = with_native(CONFIG)
        config['study']['engines'] = ['llama_cpp']
        with self.assertRaises(ValueError):
            build(config)


if __name__ == '__main__':
    unittest.main()

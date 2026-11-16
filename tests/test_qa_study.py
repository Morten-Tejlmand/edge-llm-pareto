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


def build(config):
    return qa_study.build_commands(config, config_path=CONFIG_PATH, qa_dataset='qa.jsonl')


def rows(commands):
    return [(c[c.index('--model') + 1], c[c.index('--engine') + 1], c[c.index('--rung') + 1])
            for c in commands]


class StudyMatrixTest(unittest.TestCase):
    def test_config_includes_native_vllm_rows(self):
        commands, configurations, trials = build(CONFIG)
        self.assertIn(('qwen3.8-27b', 'vllm', 'native-bf16,official-fp8'), rows(commands))
        self.assertIn(('deepseek-r1-0528-qwen3-8b', 'vllm', 'native-bf16'), rows(commands))
        # 2 models x 3 GGUF rungs x 2 engines x 5 levels + 3 native rungs x 5 levels.
        self.assertEqual(configurations, 60 + 15)
        self.assertEqual(trials, 75 * 30)

    def test_native_rows_are_optional(self):
        config = copy.deepcopy(CONFIG)
        del config['study']['native_vllm_rungs']
        commands, configurations, _ = build(config)
        self.assertEqual(configurations, 60)
        self.assertEqual(len(commands), 4)

    def test_rejects_native_rung_missing_from_vllm(self):
        config = copy.deepcopy(CONFIG)
        config['study']['native_vllm_rungs'] = {'deepseek-r1-0528-qwen3-8b': ['official-fp8']}
        with self.assertRaises(ValueError):
            build(config)

    def test_rejects_native_rows_duplicating_gguf_matrix(self):
        config = copy.deepcopy(CONFIG)
        config['study']['native_vllm_rungs'] = {'qwen3.8-27b': ['gguf-q8_0']}
        with self.assertRaises(ValueError):
            build(config)

    def test_rejects_native_rows_without_vllm_engine(self):
        config = copy.deepcopy(CONFIG)
        config['study']['engines'] = ['llama_cpp']
        with self.assertRaises(ValueError):
            build(config)


if __name__ == '__main__':
    unittest.main()

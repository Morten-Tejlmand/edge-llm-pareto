"""vLLM orchestration and measurement regressions without a GPU or downloads."""
import json
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, Mock

import test_llama_cpp
from edge_llm_pareto import bench, hardware, serve
from edge_llm_pareto.analysis import aggregate
from edge_llm_pareto.perplexity import attach_results


class VllmTests(unittest.TestCase):
    def test_failed_or_unavailable_output_tokenizer_check_blocks_trials(self):
        for result in (dict(valid=False, input_cases=0), RuntimeError('server unsupported')):
            options = dict(side_effect=result) if isinstance(result, Exception) else dict(return_value=result)
            with self.subTest(result=result), tempfile.TemporaryDirectory() as d, \
                 patch('edge_llm_pareto.tokenizer_validation.check_running_server', AsyncMock(**options)):
                (rc, row, _, run, _, _), _ = self.run_backend(d, extra=['--validate-output-tokenizer'])
                self.assertEqual(rc, 1)
                self.assertTrue(row['skipped'])
                self.assertIn('output_tokenizer_validation_failed', row['reason'])
                run.assert_not_awaited()

    def test_successful_output_tokenizer_check_is_saved_before_trials(self):
        with tempfile.TemporaryDirectory() as d, \
             patch('edge_llm_pareto.tokenizer_validation.check_running_server', AsyncMock(
                 return_value=dict(valid=True, input_cases=202, mismatches=[]))):
            (rc, row, _, run, _, _), _ = self.run_backend(d, extra=['--validate-output-tokenizer'])
            self.assertEqual(rc, 0)
            self.assertEqual(run.await_count, 2)
            artifact = Path(row['tokenizer_validation_artifact'])
            self.assertTrue(json.loads(artifact.read_text())['valid'])

    def test_user_speed_uses_each_request_latency_and_exact_counts(self):
        requests = [bench.request_result(True, token_count_source=source,
                    output_tokens=tokens, latency_s=latency, ttft_s=.1)
                    for source, tokens, latency in [('server', 10, 2),
                        ('server', 30, 3), ('chunks', 100, 1)]]
        client = Mock(close=AsyncMock())
        with patch('openai.AsyncOpenAI', return_value=client), patch.object(
                bench, '_one_request', AsyncMock(side_effect=requests)):
            row = asyncio.run(bench.run_concurrency_level('http://localhost/v1',
                'model', 1, bench.make_workload(), 3, monitor_gpu=False))
        self.assertEqual(row['user_output_tok_s_mean'], 7.5)
        self.assertEqual([r['user_output_tok_s'] for r in row['request_results']],
                         [5., 10., None])
        self.assertFalse(row['token_counts_exact'])

    def test_native_rows_do_not_receive_gguf_perplexity(self):
        row = dict(base_model='test', label='native-bf16', gguf=None)
        self.assertIsNone(attach_results([row], [])[0]['perplexity_evaluation'])

    def run_backend(self, directory, native=True, ready=True, revision=None, extra=()):
        cfg = json.loads(json.dumps(__import__('yaml').safe_load(
            (test_llama_cpp.ROOT / 'configs/models.yaml').read_text())))
        model = cfg['models']['qwen3-4b-2507']
        model['engine_rungs']['vllm'] = (dict(repo='Qwen/test', default_rung='native-bf16',
            rungs=[dict(rung='native-bf16', linear_bits=16)]) if native else
            model['engine_rungs']['llama_cpp'])
        if revision:
            model['engine_rungs']['vllm']['revision'] = revision
            model['shape_revision'] = 'shape-sha'
        path = Path(directory) / 'config.yaml'
        path.write_text(json.dumps(cfg))
        with patch.object(serve, 'resolve_hf_revision', return_value='pinned-native') as resolve:
            result = test_llama_cpp.LlamaSweepTests().run_cli(directory, ready=ready,
                extra=['--engine', 'vllm', '--config', str(path), '--repeats', '2', *extra])
        return result, resolve

    def test_native_weights_revision_memory_capacity_and_repeats(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, row, start, run, download, handle), resolve = self.run_backend(d)
        self.assertEqual(rc, 0)
        download.assert_not_called()
        resolve.assert_called_once_with('Qwen/test', 'main')
        cmd = start.call_args.args[0]
        self.assertEqual(cmd[1:3], ['serve', 'Qwen/test'])
        self.assertEqual(cmd[cmd.index('--revision') + 1], 'pinned-native')
        self.assertEqual(cmd[cmd.index('--max-num-seqs') + 1], '2')
        self.assertEqual(cmd[cmd.index('--max-model-len') + 1], '4096')
        self.assertEqual(int(cmd[cmd.index('--kv-cache-memory-bytes') + 1]), 1000 * 1024**2)
        self.assertEqual(run.await_count, 2)
        self.assertEqual(row['engine'], 'vllm')
        self.assertEqual(row['model_revision'], 'pinned-native')
        self.assertEqual(row['memory_policy'], 'explicit_kv_bytes')
        self.assertIsNone(row['gguf'])
        self.assertIsNone(row['llama_cpp_version'])
        self.assertIn('--disable-prefix-caching', cmd)
        self.assertNotIn('--enforce-eager', cmd)
        handle.__exit__.assert_called_once()

    def test_gguf_pins_separate_tokenizer_and_uses_downloaded_file(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, row, start, _, download, _), resolve = self.run_backend(d, native=False)
        self.assertEqual(rc, 0)
        download.assert_called_once()
        resolve.assert_called_once_with('unsloth/Qwen3-4B-Instruct-2507', 'main')
        cmd = start.call_args.args[0]
        self.assertEqual(cmd[2], '/cache/test.gguf')
        self.assertEqual(cmd[cmd.index('--quantization') + 1], 'gguf')
        self.assertEqual(cmd[cmd.index('--tokenizer-revision') + 1], 'pinned-native')
        # vLLM fetches the base config with --revision: it must be the tokenizer
        # repo's commit, never the GGUF repo's (which 404s in the base repo).
        self.assertEqual(cmd[cmd.index('--revision') + 1], 'pinned-native')
        self.assertEqual(row['gguf']['revision'], 'abc')
        self.assertEqual(row['model_revision'], 'abc')

    def test_native_startup_failure_is_recorded_and_process_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, row, _, run, _, handle), _ = self.run_backend(d, ready=False)
        self.assertEqual(rc, 1)
        self.assertEqual(row['reason'], 'server_not_ready')
        run.assert_not_awaited()
        handle.__exit__.assert_called_once()

    def test_nested_text_config_uses_language_dimensions(self):
        cfg = dict(num_attention_heads=8, num_hidden_layers=4, hidden_size=128,
                   vocab_size=100, intermediate_size=256)
        self.assertEqual(hardware.ModelShape.from_config(cfg, 'test'),
                         hardware.ModelShape.from_config({'text_config': cfg}, 'test'))

    def test_variability_uses_sample_sd_and_trial_count(self):
        base = dict(config_id='one', energy_valid=True, token_counts_exact=True,
                    failed=0, output_tok_s=10., latency_s_p95=1., ttft_s_p95=.1,
                    joules_per_output_token=2., run_energy_j=20.)
        rows = [dict(base, trial_id=str(i), output_tok_s=v, user_output_tok_s_mean=v)
                for i, v in enumerate([10., 20., 30.])]
        result = aggregate(rows)[0]
        self.assertEqual(result['trials'], 3)
        self.assertEqual(result['output_tok_s'], 20.)
        self.assertEqual(result['output_tok_s_std'], 10.)
        self.assertEqual(result['output_tok_s_cv_pct'], 50.)
        self.assertEqual(result['user_output_tok_s_mean_samples'], 3)
        single = aggregate(rows[:1])[0]
        self.assertIsNone(single['output_tok_s_std'])
        self.assertIsNone(single['output_tok_s_ci_low'])


    def test_configured_revision_pins_native_weights(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, *_), resolve = self.run_backend(d, revision='config-sha')
        self.assertEqual(rc, 0)
        resolve.assert_called_once_with('Qwen/test', 'config-sha')

    def test_cli_revision_overrides_configured_revision(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, *_), resolve = self.run_backend(d, revision='config-sha',
                                                 extra=['--model-revision', 'cli-sha'])
        self.assertEqual(rc, 0)
        resolve.assert_called_once_with('Qwen/test', 'cli-sha')

    def test_gguf_uses_configured_repo_and_tokenizer_revisions(self):
        with tempfile.TemporaryDirectory() as d:
            (rc, _, _, _, download, _), resolve = self.run_backend(
                d, native=False, revision='gguf-sha')
        self.assertEqual(rc, 0)
        self.assertEqual(download.call_args.args[2], 'gguf-sha')
        resolve.assert_called_once_with('unsloth/Qwen3-4B-Instruct-2507', 'shape-sha')


class TokenizerOverrideTests(unittest.TestCase):
    def test_override_copies_files_and_replaces_only_class(self):
        import json
        from unittest import mock
        from edge_llm_pareto import serve
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'; src.mkdir()
            (src / 'tokenizer.json').write_text('{"model": {"type": "BPE"}}')
            (src / 'config.json').write_text('{"model_type": "qwen3"}')
            (src / 'tokenizer_config.json').write_text(json.dumps(
                {'tokenizer_class': 'LlamaTokenizerFast', 'chat_template': 'T', 'bos_token': 'B'}))
            fetch = mock.Mock(side_effect=lambda repo, name, revision: str(src / name))
            with mock.patch('huggingface_hub.hf_hub_download', fetch):
                out = serve.prepare_tokenizer_override('org/model', 'abcdef1234567890',
                                                      'PreTrainedTokenizerFast', root=Path(d) / 'out')
            cfg = json.loads((Path(out) / 'tokenizer_config.json').read_text())
            self.assertEqual(cfg, {'tokenizer_class': 'PreTrainedTokenizerFast', 'chat_template': 'T', 'bos_token': 'B'})
            self.assertEqual((Path(out) / 'tokenizer.json').read_text(), '{"model": {"type": "BPE"}}')
            self.assertEqual((Path(out) / 'config.json').read_text(), '{"model_type": "qwen3"}')
            self.assertIn('abcdef123456', out)
            fetch.assert_any_call('org/model', 'tokenizer.json', revision='abcdef1234567890')
            # A local tokenizer directory is passed without a Hub revision.
            cmd = serve.vllm_server_command('org/model', revision='abcdef1234567890', alias='a', tokenizer=out)
            self.assertEqual(cmd[cmd.index('--tokenizer') + 1], out)
            self.assertNotIn('--tokenizer-revision', cmd)

    def test_study_config_overrides_deepseek_native_tokenizer(self):
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs' / 'study.yaml').read_text())
        rungs = cfg['models']['deepseek-r1-0528-qwen3-8b']['engine_rungs']['vllm']['rungs']
        native = next(r for r in rungs if r['rung'] == 'native-bf16')
        self.assertEqual(native['tokenizer_class_override'], 'PreTrainedTokenizerFast')

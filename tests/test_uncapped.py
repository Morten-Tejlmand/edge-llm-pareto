"""Independent performance and QA budgets retain termination diagnostics."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import bench, quality, serve


class UncappedTests(unittest.TestCase):
    def test_stream_omits_budgets_and_preserves_length_finish_reason(self):
        async def stream():
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content='answer'), finish_reason=None)])
            yield SimpleNamespace(usage=SimpleNamespace(completion_tokens=5, prompt_tokens=2),
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason='length')])
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=stream())
        result = asyncio.run(bench._one_request(client, 'test', bench.make_workload(max_output_tokens=None), 0))
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn('max_tokens', kwargs)
        self.assertNotIn('max_completion_tokens', kwargs)
        self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(result['output_tokens'], 5)
        self.assertIsNone(bench.make_workload(max_output_tokens=None)['max_output_tokens'])

    def test_performance_default_and_override_are_sent_to_server(self):
        async def stream():
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content='answer'), finish_reason='stop')])
        for workload, expected in ((bench.make_workload(), 256),
                                   (bench.make_workload(max_output_tokens=512), 512)):
            with self.subTest(expected=expected):
                client = Mock()
                client.chat.completions.create = AsyncMock(return_value=stream())
                asyncio.run(bench._one_request(client, 'test', workload, 0))
                self.assertEqual(client.chat.completions.create.call_args.kwargs['max_tokens'], expected)
                self.assertEqual(workload['output_limit_policy'], 'explicit_cap_or_eos_or_context')
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                bench.make_workload(max_output_tokens=invalid)

    def test_qa_has_no_hidden_cap_and_records_uncapped_protocol(self):
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='Paris'), finish_reason='stop')]))
        manager = Mock(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock())
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'qa.jsonl'
            output = Path(directory) / 'qa.json'
            dataset.write_text(json.dumps(dict(id='1', prompt='Capital?', answers=['Paris'])))
            args = SimpleNamespace(dataset=str(dataset), output=str(output), model='test',
                base_url='http://localhost/v1', model_repo='test', model_revision='sha',
                base_model='test', engine='llama_cpp', rung='bf16', scorer='exact-match')
            with patch('openai.AsyncOpenAI', return_value=manager):
                self.assertEqual(asyncio.run(quality.evaluate(args)), 0)
            record = json.loads(output.read_text())
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn('max_tokens', kwargs)
        self.assertNotIn('max_completion_tokens', kwargs)
        self.assertIsNone(record['max_tokens'])
        self.assertEqual(record['output_limit_policy'], 'eos_or_context')
        self.assertEqual(record['results'][0]['finish_reason'], 'stop')

    def test_llama_server_explicitly_disables_prediction_cap(self):
        command = serve.llama_server_command('test.gguf')
        self.assertEqual(command[command.index('--predict') + 1], '-1')
        self.assertIn('--no-context-shift', command)

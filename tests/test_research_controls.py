import copy
import json
import tempfile
import unittest
from pathlib import Path
from edge_llm_pareto import analysis, bench
from edge_llm_pareto.perplexity import attach_results, parse_perplexity
from edge_llm_pareto.workloads import attach_prompts, excerpts
import test_experiments


class ResearchControlsTests(unittest.TestCase):
    def bundle(self):
        return dict(reference='f16', protocol={'context': 2048, 'corpus_sha256': 'corpus'}, results=[
            dict(base_model='m', label=label, perplexity=ppl,
                 gguf=dict(repo='repo', revision='sha', files=[label + '.gguf']))
            for label, ppl in [('f16', 10), ('q4', 11)]])

    def test_completed_perplexity_only(self):
        self.assertEqual(parse_perplexity('[1]9.0\nFinal estimate: PPL = 11.25 +/- 0.3'), 11.25)
        for text in ['[1]9.0', 'Final estimate: PPL = nan', 'Final estimate: PPL = 1e999']:
            with self.assertRaises(ValueError):
                parse_perplexity(text)

    def test_matching_requires_exact_artifact(self):
        bundle = self.bundle()
        row = copy.deepcopy(bundle['results'][1])
        attached = attach_results([row], [bundle])[0]['perplexity_evaluation']
        self.assertAlmostEqual(attached['perplexity_increase_pct'], 10)
        row['gguf']['revision'] = 'different'
        self.assertIsNone(attach_results([row], [bundle])[0]['perplexity_evaluation'])
        bundle['results'].pop(0)
        with self.assertRaises(ValueError):
            attach_results([], [bundle])

    def test_fixed_prompts_are_stable_and_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prompts.jsonl'
            tasks = excerpts('one two three four five six', words=3, count=2)
            p.write_text('\n'.join(json.dumps(t) for t in tasks))
            w = attach_prompts(bench.make_workload(), p)
            self.assertEqual(bench.make_prompt(w, 0), tasks[0]['prompt'])
            self.assertEqual(bench.make_prompt(w, 2), tasks[0]['prompt'])
            self.assertIsNone(w['prompt_tokens'])
            p.write_text(json.dumps(tasks[0]) + '\n' + json.dumps(tasks[0]))
            with self.assertRaises(ValueError):
                attach_prompts(w, p)

    def test_combined_models_preserve_workload_boundaries(self):
        factory = test_experiments.AnalysisTests()
        a, b = factory.row(), factory.row('b', '2', 1, 100)
        b['base_model'] = 'larger'
        self.assertEqual(len(analysis.frontier(analysis.aggregate([a, b]))), 2)
        self.assertEqual(len(analysis.frontier(analysis.aggregate([a, b], True))), 1)
        b['workload'] = {'prompt_tokens': 99}
        self.assertEqual(len(analysis.frontier(analysis.aggregate([a, b], True))), 2)

    def test_export_filters_perplexity_and_labels_quality_basis(self):
        factory = test_experiments.AnalysisTests()
        bundle = self.bundle()
        a, b = factory.row(), factory.row('b', '2', 1, 100)
        a.update(bundle['results'][0])
        b.update(bundle['results'][1])
        rows = attach_results([a, b], [bundle])
        with tempfile.TemporaryDirectory() as d:
            result = analysis.export(rows, d, max_perplexity_increase_pct=5, combine_models=True)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['label'], 'f16')
            self.assertEqual(result[0]['quality_basis'], 'relative_compression_loss_only')

class PerplexityRunnerTests(unittest.TestCase):
    def test_native_evaluation_reference_and_artifact_bundle(self):
        import importlib.util
        import sys
        from unittest.mock import patch
        from edge_llm_pareto import serve
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('ppl_cli', root / 'scripts/perplexity.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        def download(repo, filename, revision):
            return dict(repo=repo, files=[filename], revision='resolved', path='/cache/' + filename)
        def native(command, stdout, **kwargs):
            stdout.write('Final estimate: PPL = 10.0 +/- 0.1\n')
        with tempfile.TemporaryDirectory() as d:
            corpus, output = Path(d) / 'wiki.txt', Path(d) / 'ppl.json'
            corpus.write_text('evaluation corpus')
            with patch.object(sys, 'argv', ['perplexity.py', '--model', 'qwen3-4b-2507',
                    '--rung', 'gguf-q4_k_m', '--reference', 'gguf-f16',
                    '--corpus', str(corpus), '--output', str(output)]), \
                    patch.object(serve, 'resolve_executable', return_value='llama-perplexity'), \
                    patch.object(serve, 'executable_version', return_value='build'), \
                    patch.object(serve, 'download_gguf', side_effect=download) as downloads, \
                    patch.object(cli.subprocess, 'run', side_effect=native):
                cli.main()
            bundle = json.loads(output.read_text())
            self.assertEqual(len(bundle['results']), 2)
            self.assertEqual(downloads.call_args_list[1].args[2], 'resolved')
            self.assertEqual(bundle['results'][0]['perplexity'], 10)
            self.assertEqual(bundle['protocol']['context'], 2048)


if __name__ == '__main__':
    unittest.main()

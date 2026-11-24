"""Protect procedure selection against the inspection CLI's Selective defaults."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PaperLauncherTests(unittest.TestCase):
    def test_all_six_archived_pairings_without_model_calls(self):
        expected = {
            'workflow_full_seq': ('v4.1', False),
            'workflow_conservative': ('v5', False),
            'workflow_full_sched': ('v4.1-flash', False),
            'workflow_selective': ('v6.3-i', True),
            'trajectory_r2_selective': ('v6.3-i', True),
            'decision_selective': ('v6.3-i', True),
        }
        for name, (version, selective) in expected.items():
            with self.subTest(configuration=name):
                result = subprocess.run([
                    sys.executable, str(ROOT / 'run_paper_pipeline.py'),
                    '--configuration', name, '--input-root', '/unused/images',
                    '--input-jsonl', '/unused/images.jsonl', '--output-root', '/unused/output',
                    '--model', 'paper-critic', '--dry-run',
                ], check=True, capture_output=True, text=True, timeout=120)
                settings = json.loads(result.stdout)['effective_settings']
                self.assertEqual(settings['pipeline_version'], version)
                self.assertEqual(settings.get('v63_selective_review', False), selective)
                self.assertTrue(settings['consolidation_enabled'])
                self.assertEqual(settings['workflows'], ['adversarial_consensus'])


if __name__ == '__main__':
    unittest.main()

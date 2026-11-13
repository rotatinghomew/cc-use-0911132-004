import json
import unittest
from pathlib import Path


class AnalysisInputsTests(unittest.TestCase):
    def load_data(self):
        return json.loads((Path(__file__).resolve().parents[1] / "fixtures/analysis/input.json").read_text())

    def test_forwarded_submissions(self):
        data = self.load_data()
        self.assertTrue(any(row['forwarded_from'] for row in data['submissions']))

    def test_late_judgment(self):
        data = self.load_data()
        self.assertTrue(any(row['available_at'] > data['as_of'] for row in data['judgments']))

    def test_commitment_reference(self):
        data = self.load_data()
        self.assertTrue(bool(data['previous_issues'][0]['commitment_ids']))

    def test_source_versions_and_cutoff(self):
        data = self.load_data()
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue(data["source_versions"])
        self.assertIn("+00:00", data["as_of"])

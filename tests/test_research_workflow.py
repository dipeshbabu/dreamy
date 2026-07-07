import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prompt_suppression.directions import mean_difference_direction, projection_gap, top_direction_specs
from prompt_suppression.latex import rows_to_latex_table
from prompt_suppression.results import CandidateRecord
from prompt_suppression.robustness import robustness_summary_rows
from prompt_suppression.target_generation import logit_specs, neuron_specs, parse_int_list, write_spec


class ResearchWorkflowTests(unittest.TestCase):
    def test_parse_int_list_supports_ranges(self):
        self.assertEqual(parse_int_list("1,3-5"), [1, 3, 4, 5])

    def test_target_generation_writes_spec(self):
        targets = logit_specs([" dog"]) + neuron_specs([1], [2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            write_spec(targets, path, model_size="70m")
            data = json.loads(path.read_text())

        self.assertEqual(len(data["targets"]), 2)
        self.assertEqual(data["model_size"], "70m")

    def test_direction_math_and_top_specs(self):
        a = np.array([[2.0, 0.0], [2.0, 1.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0]])
        direction = mean_difference_direction(a, b)
        self.assertGreater(projection_gap(a, b, direction), 0)
        specs = top_direction_specs(
            [
                {"name": "x", "layer": 0, "projection_gap": 1.0, "vector_path": "x0.npy"},
                {"name": "x", "layer": 1, "projection_gap": 3.0, "vector_path": "x1.npy"},
            ],
            top_k=1,
        )
        self.assertEqual(specs[0]["layer"], 1)

    def test_robustness_summary(self):
        rows = robustness_summary_rows(
            [
                CandidateRecord(
                    "t",
                    "epo:lower",
                    0,
                    "x",
                    target=1.2,
                    xentropy=2.0,
                    extra={"base_method": "epo", "variant": "lower", "base_target": 1.0, "base_xentropy": 2.0},
                )
            ],
            target_tolerance=0.25,
        )
        self.assertEqual(rows[0]["survival_rate"], 1.0)

    def test_latex_table_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            rows_to_latex_table(
                [{"method": "epo", "best_target": -1.2345}],
                path,
                columns=["method", "best_target"],
            )
            text = path.read_text()

        self.assertIn("\\begin{table}", text)
        self.assertIn("epo", text)


if __name__ == "__main__":
    unittest.main()

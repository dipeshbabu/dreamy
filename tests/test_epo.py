import unittest

import numpy as np
import torch

from dreamy.epo import History, build_pareto_frontier, combine_score


class DummyTokenizer:
    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class EpoObjectiveTests(unittest.TestCase):
    def test_combine_score_prefers_lower_target_in_minimize_mode(self):
        target = torch.tensor([1.0, -2.0])
        xentropy = torch.tensor([0.5, 0.5])
        penalties = torch.tensor([1.0])

        scores = combine_score(target, xentropy, penalties, minimize=True)

        self.assertGreater(scores[0, 1].item(), scores[0, 0].item())

    def test_pareto_frontier_supports_minimization(self):
        history = History()
        history.ids = np.array([[[1], [2]]])
        history.target = np.array([[10.0, -10.0]])
        history.xentropy = np.array([[0.0, 0.0]])
        history.keep = np.array([[0, 1]])
        history.runtime = np.array([0.0])

        frontier = build_pareto_frontier(
            DummyTokenizer(), history, Xvs=np.array([1.0]), minimize=True
        )

        self.assertEqual(frontier.target[0], -10.0)
        self.assertEqual(frontier.text[0], "2")


if __name__ == "__main__":
    unittest.main()

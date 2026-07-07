import unittest

from prompt_suppression.cli import build_parser
from prompt_suppression.robustness import deterministic_variants
from prompt_suppression.target_specs import token_id_from_text


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == " one":
            return [1]
        return [1, 2]


class SpecAndCliTests(unittest.TestCase):
    def test_token_id_from_text_requires_single_token(self):
        self.assertEqual(token_id_from_text(DummyTokenizer(), " one"), 1)
        with self.assertRaises(ValueError):
            token_id_from_text(DummyTokenizer(), "two tokens")

    def test_deterministic_variants_are_named_and_unique(self):
        variants = deterministic_variants(" Hello   world ")
        names = [name for name, _ in variants]

        self.assertIn("original", names)
        self.assertIn("instruction_wrap", names)
        self.assertEqual(len({text for _, text in variants}), len(variants))

    def test_cli_parser_accepts_run_command(self):
        parser = build_parser()
        args = parser.parse_args(
            ["run", "--spec", "spec.json", "--out", "runs/x", "--methods", "random"]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.methods, ["random"])


if __name__ == "__main__":
    unittest.main()

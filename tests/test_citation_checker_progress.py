import unittest
from unittest.mock import patch

from citation_checker.checker import check_references
from citation_checker.models import Reference


class CitationCheckerProgressTests(unittest.TestCase):
    def test_explicit_progress_uses_tqdm_when_available(self):
        refs = [Reference(raw_text="a", title="A"), Reference(raw_text="b", title="B")]

        with patch("citation_checker.checker.tqdm", side_effect=lambda items, **_: list(items)) as mock_tqdm, \
             patch("citation_checker.checker._check_single", side_effect=["r1", "r2"]):
            results = check_references(refs, show_progress=True, progress_desc="Test")

        self.assertEqual(results, ["r1", "r2"])
        mock_tqdm.assert_called_once()

    def test_progress_not_used_when_disabled(self):
        refs = [Reference(raw_text="a", title="A")]

        with patch("citation_checker.checker.tqdm") as mock_tqdm, \
             patch("citation_checker.checker._check_single", return_value="r1"):
            results = check_references(refs, show_progress=False)

        self.assertEqual(results, ["r1"])
        mock_tqdm.assert_not_called()


if __name__ == "__main__":
    unittest.main()

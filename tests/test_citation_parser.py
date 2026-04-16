import unittest

from citation_checker.parser import parse_references
from doc_preprocess import _normalize_ref_entries


class CitationParserTests(unittest.TestCase):
    def test_extracts_neurips_title_from_wrapped_numbered_reference(self):
        text = """
[1] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant
Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. TPU v4: An optically reconfig-
urable supercomputer for machine learning with hardware support for embeddings. In Annual
International Symposium on Computer Architecture, pages 1–14, 2023.
"""
        ref = parse_references(text)[0]
        self.assertEqual(
            ref.title,
            "TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings",
        )

    def test_extracts_icml_journal_style_title(self):
        text = """
[1] Gupta, V., Fernandez-Crehuet, J. M., and Hanne, T. Freelancers in the software development
process: A systematic mapping study. Processes, 8(10):1215, 2020. doi: 10.3390/pr8101215.
URL https://www.mdpi.com/2227-9717/8/10/1215.
"""
        ref = parse_references(text)[0]
        self.assertEqual(
            ref.title,
            "Freelancers in the software development process: A systematic mapping study",
        )

    def test_does_not_misquote_author_block_in_author_year_reference(self):
        entry = (
            "Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. "
            "In International Conference on Learning Representations, 2015."
        )
        normalized = _normalize_ref_entries([entry])[0]
        self.assertEqual(normalized, f"[1] {entry}")

        ref = parse_references(normalized)[0]
        self.assertEqual(ref.title, "Adam: A Method for Stochastic Optimization")

    def test_recovers_title_from_legacy_misquoted_reference(self):
        text = """
[1] Diederik P. "Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization". In
*International Conference on Learning Representations*, 2015.
"""
        ref = parse_references(text)[0]
        self.assertEqual(ref.title, "Adam: A Method for Stochastic Optimization")

    def test_extracts_title_before_arxiv_preprint_venue(self):
        text = """
[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
"""
        ref = parse_references(text)[0]
        self.assertEqual(ref.title, "Proximal policy optimization algorithms")

    def test_extracts_title_before_year_and_url(self):
        text = """
[1] Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E.,
Cai, C., Terry, M., Le, Q., and Sutton, C. Program synthesis with large language models,
2021. URL https://arxiv.org/abs/2108.07732.
"""
        ref = parse_references(text)[0]
        self.assertEqual(ref.title, "Program synthesis with large language models")


if __name__ == "__main__":
    unittest.main()

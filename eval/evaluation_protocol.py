"""Immutable embedding identity for the trigger screening metric.

Revision verified against the public model repository with git ls-remote.
Changing these constants before a new experiment is allowed; reusing an old
sealed experiment with a different metric/model is rejected by the evaluator.
"""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

"""
ai_engine/clause_classifier/__init__.py

Public surface of the clause_classifier package.

Usage (from FastAPI backend or any other caller):

    from clause_classifier import predict_clause, predict_batch_clauses

    result  = predict_clause("The agreement may be terminated with 30 days notice.")
    results = predict_batch_clauses(["para one ...", "para two ..."])
"""

from .predict import predict_clause, predict_batch_clauses  # noqa: F401

__all__ = ["predict_clause", "predict_batch_clauses"]

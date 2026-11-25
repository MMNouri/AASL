"""AASL: AI-Augmented Supervised Learning.

A deductive-inductive-deductive procedure for coding text with a language
model against a theory-derived category schema (Algorithm 1).
"""
from .core import (OUT_OF_DOMAIN, RESIDUAL, AwaitingHumanInput, CalibrationFailed, Coding,
                   ConfirmationFailed, Label, ReliabilityTooLow, TextUnit)
from .pipeline import AASLRun, RunResult, Settings
from .schema import Attribute, Category, Domain, Schema

__version__ = "0.1.1"

__all__ = ["AASLRun", "RunResult", "Settings", "Schema", "Domain", "Category", "Attribute",
           "TextUnit", "Label", "Coding", "RESIDUAL", "OUT_OF_DOMAIN", "AwaitingHumanInput",
           "CalibrationFailed", "ConfirmationFailed", "ReliabilityTooLow", "__version__"]

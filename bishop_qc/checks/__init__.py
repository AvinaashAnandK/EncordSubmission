"""Checks, one module per category. Importing a module registers its checks."""
from . import attribution, boundary, completeness, faithfulness, granularity, navigation, nonaction  # noqa: F401
from .runner import REGISTRY, CheckContext, CheckResult, CheckSpec, evidence, run_check

__all__ = ["REGISTRY", "CheckContext", "CheckResult", "CheckSpec", "evidence", "run_check"]

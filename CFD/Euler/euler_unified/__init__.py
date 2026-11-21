"""Unified one-dimensional Euler solver (FV/DG, single/two-medium).

The public entry point is :func:`euler_unified.run.run_case`; see
``run_euler.py`` for the command-line interface.
"""

__all__ = ["config", "eos", "initial", "numerics", "single", "two_medium", "run"]

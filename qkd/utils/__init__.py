# qkd/utils/__init__.py

# -*- coding: utf-8 -*-
"""
Utilities sub-package for the QKD simulation framework.

This package contains small, reusable helper functions for tasks like
mathematical calculations and parameter validation.
"""

from .math import (
    binary_entropy,
    hoeffding_bounds,
    clopper_pearson_bounds,
    ConfidenceInterval,
    IntervalMethod,
    calculate_total_click_probability,
)

__all__ = [
    "binary_entropy",
    "hoeffding_bounds",
    "clopper_pearson_bounds",
    "ConfidenceInterval",
    "IntervalMethod",
    "calculate_total_click_probability",
]

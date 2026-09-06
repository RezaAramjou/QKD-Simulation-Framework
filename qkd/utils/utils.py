# -*- coding: utf-8 -*-
"""
Core utility functions for the QKD simulation framework.

This module contains shared, independent helper functions that do not have
dependencies on other modules within the framework, thereby preventing
circular import issues.
"""
import json

__version__ = "1.0.0"

__all__ = ["sanitize_for_serialization"]


def sanitize_for_serialization(value: any) -> any:
    """
    Recursively sanitizes a value to ensure it is JSON-serializable and picklable.
    Converts numpy scalars to Python natives, and stringifies unknown objects.
    """
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy and isinstance(value, np.generic):
        if np.isnan(value):
            return None
        return value.item()
    
    if has_numpy and isinstance(value, np.ndarray):
        return sanitize_for_serialization(value.tolist())

    if isinstance(value, float) and value != value:
        return None

    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): sanitize_for_serialization(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_serialization(item) for item in value]

    try:
        json.dumps(value)
        return value
    except (TypeError, OverflowError):
        if isinstance(value, float) and value != value: 
             return None
        return str(value)

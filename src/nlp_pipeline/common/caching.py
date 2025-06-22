"""
Caching Module

This module provides a decorator for caching the outputs of functions using joblib.Memory.
The cache directory is defined in the configuration constants (CACHE_DIR).

Configuration:
    - CACHE_DIR: The directory path where cached outputs are stored, imported from nlp_pipeline.config.constants.

Usage Example:
    @cached
    def expensive_computation(x, y):
         # perform computation
         return result
"""

from typing import Any, Callable, TypeVar

import joblib

from nlp_pipeline.config.constants import CACHE_DIR

# Type variable for a callable function
F = TypeVar("F", bound=Callable[..., Any])

# Initialise joblib.Memory with the specified cache directory
memory = joblib.Memory(location=CACHE_DIR, verbose=0)

# --------------------------------------------------------------------------- #
# Caching Func                                                                #
# --------------------------------------------------------------------------- #

def cached(func: F) -> F:
    """
    Decorator to cache function outputs using joblib.Memory.

    This decorator caches the output of the decorated function based on its input arguments.
    It leverages joblib's Memory caching mechanism and stores cache in the directory specified
    by CACHE_DIR.

    Args:
        func (Callable[..., Any]): The function to be cached.

    Returns:
        Callable[..., Any]: A wrapped version of the original function with caching enabled.
    """
    return memory.cache(func)

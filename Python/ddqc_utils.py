#!/usr/bin/env python3

import numpy as np

def mad(x: np.ndarray, constant: float = 1.4826) -> float:
    """Function that computes adjusted MAD for numpy array"""
    return constant * np.median(np.absolute(x - np.median(x)))
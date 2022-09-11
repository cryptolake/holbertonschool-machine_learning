#!/usr/bin/env python3
"""Get F1 score from confusion matrix."""
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Get F1 score from confusion matrix."""
    pres = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * ((pres * sens) / (pres + sens))

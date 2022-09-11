#!/usr/bin/env python3
"""Get Sensitivity form confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """Get Sensitivity from confusion matrix."""
    classes = confusion.shape[0]
    sens = np.zeros((classes))
    for cl in range(classes):
        TP = confusion[cl, cl]
        N = np.sum(confusion[cl, :])
        sens[cl] = TP/N
    return sens

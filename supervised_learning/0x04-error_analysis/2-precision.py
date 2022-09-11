#!/usr/bin/env python3
"""Get Precision form confusion matrix."""
import numpy as np


def precision(confusion):
    """Get Precision from confusion matrix."""
    classes = confusion.shape[0]
    pres = np.zeros((classes))
    for cl in range(classes):
        TP = confusion[cl, cl]
        R = np.sum(confusion[:, cl])
        pres[cl] = TP/R
    return pres

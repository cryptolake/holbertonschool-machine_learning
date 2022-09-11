#!/usr/bin/env python3
"""Get Specifity from confusion matrix."""
import numpy as np


def specificity(confusion):
    """Get Specifity from confusion matrix."""
    classes = confusion.shape[0]
    spec = np.zeros((classes))
    all = np.sum(confusion)
    for cl in range(classes):
        # true positive
        TP = confusion[cl, cl]
        # false negative
        FN = np.sum(confusion[cl, :]) - TP
        # false positive
        FP = np.sum(confusion[:, cl]) - TP
        # True negative
        TN = all - (FN + TP + FP)
        spec[cl] = TN/(TN + FP)
    return spec

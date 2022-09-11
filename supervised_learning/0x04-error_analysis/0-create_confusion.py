#!/usr/bin/env python3
"""Confusion Matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create Confusion Matrix."""
    cm = np.zeros((labels.shape[1], labels.shape[1]))
    m = labels.shape[0]
    for i in range(m):
        inda = np.argwhere(labels[i] == 1)
        indo = np.argwhere(logits[i] == 1)
        cm[inda, indo] += 1
    # Another way of doing this is
    # np.matmul(labels.T, logits)
    return cm

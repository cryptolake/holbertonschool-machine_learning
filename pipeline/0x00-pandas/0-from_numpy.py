#!/usr/bin/env python3
"""From numpy array to pandas."""
import pandas as pd

def from_numpy(array):
    """Get pandas df from numpy array."""

    df = pd.DataFrame(array, columns=[chr(65+x) for x in range(array.shape[1])])
    return df

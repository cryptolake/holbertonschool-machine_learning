#!/usr/bin/env python3
"""Pandas df from file."""
import pandas as pd

def from_file(filename, delimiter):
    """Pandas df from file."""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df

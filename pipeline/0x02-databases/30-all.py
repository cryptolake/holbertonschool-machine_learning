#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo


def list_all(mongo_collection):
    """List all documents."""
    return list(mongo_collection.find({}))

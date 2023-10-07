#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo


def insert_school(mongo_collection, **kwargs):
    """Insert document."""
    inserted = mongo_collection.insert_one(kwargs)
    return inserted.inserted_id

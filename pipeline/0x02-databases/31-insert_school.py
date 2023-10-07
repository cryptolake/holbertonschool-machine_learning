#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo


def insert_school(mongo_collection, **kwargs):
    """Insert document."""
    return str(mongo_collection.insert_one(kwargs).inserted_id)

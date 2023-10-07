#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo


def update_topics(mongo_collection, name, topics):
    """Update collection."""
    mongo_collection.update_many({'name': name}, {"$set": {"topics": topics}})

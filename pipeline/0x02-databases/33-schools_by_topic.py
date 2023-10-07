#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo


def schools_by_topic(mongo_collection, topic):
    """Advanced find."""
    return mongo_collection.find({"topics": {"$in": [topic]}})

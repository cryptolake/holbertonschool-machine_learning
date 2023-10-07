#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo

def top_students(mongo_collection):
    """Top students by avg score."""
    return mongo_collection.aggregate([
        {"$group": {"_id": "$name", "averageScore": {"$avg": "$score"}}},
        {"$sort": {"averageScore": -1}}
    ])

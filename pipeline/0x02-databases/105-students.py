#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo

def top_students(mongo_collection):
    """Top students by avg score."""
    return mongo_collection.aggregate([
        {"$unwind": "$topics"},
        {"$group": {"_id": {
            "_id": "$_id",
            "name": "$name"
        },
                    "averageScore": {"$avg": "$topics.score"}}},
        {"$set": {"name": "$_id.name", "_id":"$_id._id"}},
        {"$sort": {"averageScore": -1}},
    ])

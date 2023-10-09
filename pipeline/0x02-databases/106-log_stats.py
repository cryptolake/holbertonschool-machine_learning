#!/usr/bin/env python3
"""Mongodb in python."""
import pymongo as pg

def count_method(collection, method):
    """Count method from nginx logs."""
    return collection.count_documents({"method": method})


if __name__ == "__main__":
    client = pg.MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    print("{} logs".format(nginx.count_documents({})))
    print("Methods:")
    for method in methods:
        print("\tmethod {}: {}".format(method, count_method(nginx, method)))
    print("{} status check".format(nginx.count_documents({"method": "GET",
                                                          "path": "/status"})))
    top_ips = nginx.aggregate([
        {"$group": {
            "_id": "$ip",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}

    ])
    print("IPs:")
    for ip in top_ips:
        print("\t{}: {}".format(ip.get('_id'), ip.get('count')))

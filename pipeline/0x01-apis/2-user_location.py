#!/usr/bin/env python3
"""Get location of github user."""
import requests
import sys
import time


def github_location(url):
    """Get location of github user."""
    g_req = requests.get(url)
    if g_req.status_code == 403:
        reset = int(g_req.headers['X-Ratelimit-Reset'])
        tm_to_reset = reset - int(time.time())
        print("Reset in {} min".format(tm_to_reset//60))
    elif g_req.status_code == 404:
        print("Not found")
    else:
        g_user = g_req.json()
        print(g_user['location'])


if __name__ == "__main__":
    github_location(sys.argv[1])

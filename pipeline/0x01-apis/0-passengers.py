#!/usr/bin/env python3
"""Return the list of ships that hold given number of passengers."""
import requests


def availableShips(passengerCount):
    """Return the list of ships that hold given number of passengers."""
    starships_request = requests.get('https://swapi-api.hbtn.io/api/starships/').json()
    starships = starships_request['results'].copy()
    next = starships_request['next']
    while next is not None:
        starships_request = requests.get(next).json()
        starships.extend(starships_request['results'].copy())
        next = starships_request.get('next')
    available_starships = filter(lambda x: x['passengers'].isdigit() and int(x['passengers'].replace(',', '')) >= passengerCount, starships)
    return list(map(lambda x: x['name'], available_starships))

#!/usr/bin/env python3
"""Get number of rockets."""
import requests


def rocket_freq():
    """Get Rocket frequency."""
    rockets = requests.get('https://api.spacexdata.com/latest/rockets/').json()
    rockets = sorted(rockets, key=lambda x: x['name'])
    rocket_dict = {rocket['id']: [rocket['name'], 0]
                   for rocket in rockets}

    launches = requests.get('https://api.spacexdata.com/latest/launches')\
        .json()

    for launch in launches:
        rocket_dict[launch['rocket']][1] += 1

    rocket_dict = dict(sorted(rocket_dict.items(),
                              key=lambda x: x[1][1], reverse=True))

    for rocket in rocket_dict.values():
        print("{}: {}".format(rocket[0], rocket[1]))


if __name__ == "__main__":
    rocket_freq()

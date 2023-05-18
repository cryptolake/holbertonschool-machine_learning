#!/usr/bin/env python3
"""Get Upcoming SpaceX launch."""
import requests


def upcoming_launch():
    """Get Upcoming SpaceX launch."""
    launches = requests.get('https://api.spacexdata.com/latest/launches').json()
    latest = sorted(launches, key=lambda x: x['date_unix'], reverse=True)[0]

    rocket = requests.get('https://api.spacexdata.com/latest/rockets/{}'
                          .format(latest['rocket'])).json()

    launchpad = requests.get('https://api.spacexdata.com/latest/launchpads/{}'
                             .format(latest['launchpad'])).json()

    print("{} ({}) {} - {} ({})".format(latest['name'], latest['date_local'],
                                        rocket['name'],launchpad['name'],
                                        launchpad['locality']))
if __name__ == "__main__":
    upcoming_launch()

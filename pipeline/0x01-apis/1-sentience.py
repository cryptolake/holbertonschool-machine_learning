#!/usr/bin/env python3
"""Return the list of sentient planets."""
import requests


def sentientPlanets():
    """Return the list of sentient planets."""
    species = []
    next = 'https://swapi-api.hbtn.io/api/species/'

    while next is not None:
        req = requests.get(next).json()
        species.extend(req['results'])
        next = req.get('next')

    def get_planet(species):
        planet = requests.get(species['homeworld']).json()
        return planet['name']

    sentient_species = filter(lambda x: x['designation'] == 'sentient'
                              and x['homeworld'] is not None, species)
    return list(map(get_planet, sentient_species))

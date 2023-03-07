#!/usr/bin/env python3
"""
Bayesian optimization for random forest regressor applied
on housing data.
"""
from GPyOpt.methods import BayesianOptimization
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from typing import Tuple, Callable
from joblib import dump
import numpy as np


def save_model(Model: RandomForestRegressor, params: list):
    """
    Save the random forest regressor model.

    Model: model to save
    params: parameters for the random forest model file
    the name of the model will have the 5 hyperparameters with
    an - between each two of them.
    """
    dump(Model, './{}-{}-{}-{}-{}'.format(*params))


def create_model(params: list) -> RandomForestRegressor:
    """
    Create a random forest Regressor model

    args:
        params: parameters for the random forest model
        [
            0: n_estimators: number of trees
            1: max_depth: max depth of trees
            2: max_features: max number of features to consider when split
            3: max_samples: the number of samples from X to each tree
            4: max_leaf_nodes: build with the maximum leaf nodes number
        ]

    Returns:
        Model: model to fit data with
    """
    Model = RandomForestRegressor(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        max_features=params[2],
        max_samples=params[3],
        max_leaf_nodes=int(params[4])
    )

    return Model


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the california housing data and return x, y
    https://scikit-learn.org/stable/datasets/real_world.html\
    #california-housing-dataset

    Returns:
        X: features to predict from
        Y: target prediction (house price)
    """
    data = fetch_california_housing(data_home='./data')
    return data['data'], data['target']


def get_opt_func(X: np.ndarray, Y: np.ndarray) -> Callable[[list], np.ndarray]:
    """
    Create the function to evaluate.

    Args:
        X: params to predict from
        Y: target
    Return:
        f: function to optimize
    """

    def f_eval(params: list) -> np.ndarray:
        """
        Function to optimize: optimization objective
        Args:
            params: parameters for the random forest model
        Return:
            score: using corss validation (we take a different portion
                of data for testing at each iterations)
        """
        params = params[0]
        Model = create_model(params)
        score = cross_val_score(
            Model, X, Y, scoring='neg_mean_squared_error').mean()
        save_model(Model, params)
        return np.array(score)
    return f_eval


def optimize() -> None:
    """Main function when the script is run."""
    X, Y = load_data()
    domains = [
        {'name': 'n_estimators', 'type': 'discrete',
            'domain': range(64, 300)},

        {'name': 'max_depth', 'type': 'discrete',
            'domain': range(3, 100)},

        {'name': 'max_features', 'type': 'continuous',
            'domain': (0.1, 1), 'dimensionality': 1},

        {'name': 'max_samples', 'type': 'continuous',
            'domain': (0.1, 1), 'dimensionality': 1},

        {'name': 'max_leaf_nodes', 'type': 'discrete',
            'domain': range(2, 200)}
    ]
    f = get_opt_func(X, Y)
    optimizer = BayesianOptimization(f=f,
                                     domain=domains,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     exact_feval=True,
                                     acquisition_jitter=0.05,
                                     maximize=True)
    optimizer.run_optimization(max_iter=30, verbosity=True,
                               report_file='bayes_opt.txt')

    optimizer.plot_convergence()


if __name__ == '__main__':
    optimize()

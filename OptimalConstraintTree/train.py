""" Training wrappers for IAI """
import numpy as np
from interpretableai import iai

def train_trees(X, Y, **kwargs):
    """ Wrapper for IAI grid search with optional kwargs."""
    lnr_kwargs = {'regression_sparsity': 'all',
                  'fast_num_support_restarts': 3,
                  'hyperplane_config': [{'sparsity': 1}],
                  'random_seed': 314}
    split_data_kwargs = {'seed': 314,
                         'train_proportion': 0.6}
    grid_kwargs = {'max_depth': [2, 3],
                   'cp': [0.01, 0.005],
                   'regression_lambda': [0.0001]
                   }
    for key, value in kwargs.items():
        # Setting learner parameters
        if key in ['regression_sparsity', 'hyperplane_config', 'fast_num_support_restarts']:
            lnr_kwargs.update({key: value})
        elif key in ['regression_lambda', 'max_depth', 'cp']:
            grid_kwargs.update({key:value})
        elif key == 'seed':
            lnr_kwargs.update({'random_seed': value})
            split_data_kwargs.update({key: value})
        elif key == 'train_proportion':
            split_data_kwargs.update({key, value})
        else:
            raise ValueError("Kwarg with key %s is invalid." % key)

    lnr = iai.OptimalTreeRegressor()
    lnr.set_params(**lnr_kwargs)
    (train_X, train_Y), (test_X, test_Y) = iai.split_data('regression', X, Y,
                                                          **split_data_kwargs)
    grid = iai.GridSearch(lnr, **grid_kwargs)
    grid.fit(train_X, train_Y, test_X, test_Y)
    return grid

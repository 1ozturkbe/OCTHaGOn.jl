import numpy as np
from gpkit import Variable, Model
import scipy.optimize as op
import os
import interpretableai as iai

from OptimalConstraintTree.constraintify import trust_region_data, pwl_constraint_data

import unittest
from gpkit.tests.helpers import run_tests

class TestModels(unittest.TestCase):
    def test_airfoil_model(self):
        vks = ["Re", "thick", "M", "C_L"];

        # Airfoil data over Reynolds #, thickness, Mach #, and lift coeff
        X = pd.read_csv("../data/airfoil/airfoil_X.csv", header=None)
        y = pd.read_csv("../data/airfoil/airfoil_Y.csv", header=None)
        X = np.array(X);
        y = np.array(y);

        # Values of independent vars in exponential space for reference
        Re = np.linspace(10000, 35000, num=6, endpoint=True);
        thick = np.array([0.100, 0.110, 0.120, 0.130, 0.140, 0.145]);
        M = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        cl = np.linspace(0.35, 0.70, num=8, endpoint=True);

        # Splitting and training tree over data
        (train_X, train_y), (test_X, test_y) = iai.split_data('regression', X, y, seed=1)
        grid = iai.GridSearch(iai.OptimalTreeRegressor(random_seed=1, regression_sparsity='all',
                                                       hyperplane_config={'sparsity': 1},
                                                       fast_num_support_restarts=3),
                              regression_lambda=[0.001],
                              max_depth=[2], cp=[0.01, 0.05, 0.001], )
        grid.fit(train_X, train_y, test_X, test_y)
        lnr = grid.get_learner()

        # Getting trust region data
        upperDict, lowerDict = trust_region_data(lnr, vks)

        # PWL approximation data
        pwlData = pwl_constraint_data(lnr, vks)

TESTS = [TestModels]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

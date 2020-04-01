""" Tests for ORT models from data """
import numpy as np
from numpy.random import exponential as xp
from interpretableai import iai
import pandas as pd
import pickle
from gpkit import Variable
from gpkit.small_scripts import mag
from gpfit.fit import fit

from OptimalConstraintTree.constraintify import trust_region_data, pwl_constraint_data
from OptimalConstraintTree.sample import sample_gpobj

import unittest
from gpkit.tests.helpers import run_tests
from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *

class TestModels(unittest.TestCase):
    """ Test cases for different ORT models.
    Also tests constraint data generation from ORTs.
    """
    def test_airfoil_model(self):
        vks = ["Re", "thick", "M", "C_L"]

        # Airfoil data over Reynolds #, thickness, Mach #, and lift coeff
        X = pd.read_csv("../../data/airfoil/airfoil_X.csv", header=None)
        y = pd.read_csv("../../data/airfoil/airfoil_Y.csv", header=None)
        X = np.array(X)
        y = np.array(y)

        # Values of independent vars in exponential space for reference
        # Re = np.linspace(10000, 35000, num=6, endpoint=True)
        # thick = np.array([0.100, 0.110, 0.120, 0.130, 0.140, 0.145])
        # M = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # cl = np.linspace(0.35, 0.70, num=8, endpoint=True)

        # Splitting and training tree over data
        (train_X, train_y), (test_X, test_y) = iai.split_data('regression', X, y, seed=1)
        grid = iai.GridSearch(iai.OptimalTreeRegressor(regression_sparsity='all',
                                                       hyperplane_config={'sparsity': 1},
                                                       fast_num_support_restarts=3),
                              regression_lambda=[0.001],
                              max_depth=[2, 3], cp=[0.01, 0.005])
        grid.fit(train_X, train_y, test_X, test_y)
        lnr = grid.get_learner()

        # Getting trust region data
        upperDict, lowerDict = trust_region_data(lnr, vks)

        # PWL approximation data
        _ = pwl_constraint_data(lnr, vks)

        # Testing number of splits is equal to depthlh
        for key in list(upperDict.keys()):
            self.assertEqual(len(upperDict[key]) + len(lowerDict[key]), lnr.get_depth(key))

    def test_gpobj_model(self):
        # TODO: compare different sampling methods here.
        a = Variable()
        b = Variable()
        xp = np.random.exponential
        gpobjs = [a*b**(-1.2), a + b**(2.1)]
        samples = 10
        for gpobj in gpobjs:
            bounds = {a.key: [xp(), xp()], b.key: [xp(), xp()]}
            Y, X = sample_gpobj(gpobj, bounds, samples)
            X = np.log([[mag(X[i][key]) for i in range(samples)] for key in list(bounds.keys())])
            Y = np.log([y.value for y in Y])
            cstrt, rms = fit(X, Y, 2, 'SMA')
            # print('RMS: ' + str(rms))

    def test_fit_gpmodel(self):
        samples = 300
        bounds = pickle.load(open("subs/SimPleAC.bounds", "rb"))
        solns = [pickle.load(open("solns/SimPleAC" + str(i) + ".sol", "rb")) for i in range(samples)]
        subs = [pickle.load(open("subs/SimPleAC" + str(i) + ".subs", "rb")) for i in range(samples)]
        X = np.log([[mag(subs[i][key]) for i in range(samples)] for key in list(bounds.keys())])
        Y = np.log([mag(soln['cost']) for soln in solns])
        cstrt, rms = fit(X, Y, 4, 'SMA')
        # print('RMS: ' + str(rms))

TESTS = [TestModels]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

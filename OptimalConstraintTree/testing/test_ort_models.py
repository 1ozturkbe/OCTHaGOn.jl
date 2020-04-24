""" Tests for ORT models from data """
import numpy as np
from numpy.random import exponential as xp
import pandas as pd
import pickle
from gpkit.small_scripts import mag
from gpkit import VectorVariable, Variable, units, Model
from gpfit.fit import fit

from OptimalConstraintTree.constraint_tree import ConstraintTree
from OptimalConstraintTree.sample import sample_gpobj, gen_X
from OptimalConstraintTree.train import train_trees
from OptimalConstraintTree.tools import enablePrint, blockPrint, prep_SimPleAC

import unittest
from gpkit.tests.helpers import run_tests

class TestORTModels(unittest.TestCase):
    """ Test cases for different ORT models.
    Also tests constraint data generation from ORTs.
    """

    def test_train_trees(self):
        """ Tests train_trees over transonic XFOIL analysis."""
        # Airfoil data over Reynolds #, thickness, Mach #, and lift coeff
        X = pd.read_csv("../../data/airfoil/airfoil_X.csv", header=None)
        Y = pd.read_csv("../../data/airfoil/airfoil_Y.csv", header=None)
        X = X.values
        Y = Y.values.flatten()

        # Values of independent vars in exponential space for reference
        # Re = np.linspace(10000, 35000, num=6, endpoint=True)
        # thick = np.array([0.100, 0.110, 0.120, 0.130, 0.140, 0.145])
        # M = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # cl = np.linspace(0.35, 0.70, num=8, endpoint=True)

        # Fitting with GPfit
        # blockPrint()
        cstrt, rms = fit(np.transpose(X), Y, 3, 'SMA')
        # enablePrint()

        # Splitting and training tree over data (dummy config inputs for testing)
        grid = train_trees(X, Y, seed=314,
                           regression_sparsity='all',
                           fast_num_support_restarts=2,
                           regression_lambda=[0.001],
                           max_depth=[2],
                           minbucket=[0.05],
                           hyperplane_config=[{'sparsity': 1, 'feature_set': [3, 4]},
                                              {'sparsity': 2, 'feature_set': [1, 2, 4]}])
        lnr = grid.get_learner()
        lnr.write_json("data/airfoil_lnr.json")

        # Defining dummy variables of interest
        vks = ['x' + str(i) for i in range(1, 5)]
        ivars = VectorVariable(len(vks), 'x')
        dvar = Variable('y')

        # Getting trust region data
        upperDict, lowerDict = ConstraintTree.trust_region_data(lnr, vks)

        # Making constraints from trust region data
        tr_constraints = ConstraintTree.tr_constraintify(upperDict, lowerDict, ivars)

        # PWL approximation data
        pwlDict = ConstraintTree.pwl_constraint_data(lnr, vks)

        # PWL constraints
        pwl_constraints = ConstraintTree.pwl_constraintify(pwlDict, dvar, ivars)

        # Testing number of splits is equal to depth
        for key in list(upperDict.keys()):
            self.assertEqual(len(upperDict[key]) + len(lowerDict[key]), lnr.get_depth(key))
            self.assertEqual(len(tr_constraints[key]), lnr.get_depth(key))

        # Finally creating a ConstraintTree from the learner
        ct = ConstraintTree(lnr, dvar, ivars)

    def test_SimPleAC_dragmodel(self):
        """ Tests train_trees over the SimPleAC/solar airfoils.
        Note: solar/berkRobust branch is the branch where the fits are best! """
        re_range = np.array([500, 1000, 1500, 2000, 2500, 3000])*1000
        tau_range = np.array([80, 90, 100, 110, 120, 130, 140, 150,
                              160, 170, 180, 190, 200, 210, 220, 230])/1000.
        re_ref = 1500000.
        tau_ref = 120./1000.
        X = pickle.load(open("../../data/airfoil/solar.X", "rb")) # [CL, Re, tau]
        Y = pickle.load(open("../../data/airfoil/solar.Y", "rb")) # [CD]

        # Splitting and training tree over data
        grid = train_trees(np.transpose(X), Y, seed=314,
                           regression_sparsity='all',
                           fast_num_support_restarts=2,
                           regression_lambda=[0.00001],
                           max_depth=[2],
                           minbucket=[0.05],
                           hyperplane_config=[{'sparsity': 1}])
        lnr = grid.get_learner()
        lnr.write_json("data/solar_airfoil_lnr.json") # Saving for later use

    def test_gpobj_model(self):
        """ Tests surrogate models over GPkit objects. """
        # TODO: compare different sampling methods here.
        a = Variable()
        b = Variable()
        xp = np.random.exponential
        gpobjs = [a * b ** (-1.2), a + b ** (2.1)]
        samples = 20
        for gpobj in gpobjs:
            bounds = {a.key: [xp(), xp()], b.key: [xp(), xp()]}
            basis = {a.key: 1, b.key: 1}
            res, subs = sample_gpobj(gpobj, bounds, samples)
            Y = [r.value for r in res]
            X = gen_X(subs, basis)
            # GP fit
            # blockPrint()
            cstrt, rms = fit(np.log(np.transpose(X)), np.log(Y), 2, 'SMA')
            # enablePrint()
            self.assertAlmostEqual(rms, 0, places=5)
            # Tree fit
            grid = train_trees(np.log(X), np.log(Y), max_depth=5)
            lnr = grid.get_learner()
            self.assertAlmostEqual(lnr.score(np.log(X), np.log(Y)), 1, places=2)

    def test_fit_gpmodel(self):
        """ Computes and compares posynomial surrogate
        for GPmodel with actual model. """
        m, basis = prep_SimPleAC()
        # blockPrint()
        basesol = m.localsolve(verbosity=0)
        # enablePrint()
        solns = pickle.load(open("data/SimPleAC.sol", "rb"))
        subs = pickle.load(open("data/SimPleAC.subs", "rb"))
        X = gen_X(subs, basis)
        Y = [mag(soln['cost'] / basesol['cost']) for soln in solns]

        # GPfitted model with generated constraint
        # blockPrint()
        cstrt, rms = fit(np.log(np.transpose(X)), np.log(Y), 4, 'SMA')
        # enablePrint()
        # self.assertAlmostEqual(rms, 0.03, places=0)

        # ML model
        grid = train_trees(np.log(X), np.log(Y),
                           fast_num_support_restarts=2,
                           regression_lambda=[0.00001],
                           max_depth=[3],
                           minbucket=[0.01])
        lnr = grid.get_learner()
        lnr.write_json("data/SimPleAC_lnr.json")
        self.assertAlmostEqual(lnr.score(np.log(X), np.log(Y)), 1, places=2)

TESTS = [TestORTModels]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

import numpy as np
from gpkit import Variable, Model
from gpkit.small_scripts import mag
from gpfit.fit import fit

import unittest
from gpkit.tests.helpers import run_tests

from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *
from interpretableai import iai
import pickle

from OptimalConstraintTree.constraint_tree import ConstraintTree
from OptimalConstraintTree.global_model import GlobalModel
from OptimalConstraintTree.sample import gen_X
from OptimalConstraintTree.tools import (find_signomials, prep_SimPleAC,
                                        get_varkeys, get_bounds, \
                                        constraints_from_bounds, \
                                        constraint_from_gpfit)

class TestConstraintify(unittest.TestCase):
    def test_monomials_from_pwl_data(self):
        pwlDict = {1: [1, [2,3,4]],
                   2: [5, [6,7,8]],
                   3: [9, [10,11,12]]}
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        ivars = [a, b*c**0.35, c]
        dvar = a*b
        constraintDict = ConstraintTree.pwl_constraintify(pwlDict, dvar, ivars)
        test_constr = np.exp(1)*ivars[0]**2*ivars[1]**3*ivars[2]**4 <= dvar
        self.assertEqual(test_constr.as_hmapslt1({}),
                         constraintDict[1][0].as_hmapslt1({}))

        # Testing tree

    def test_ConstraintTree_sp_constraints(self):
        """
        Tests ConstraintTree generation from SP constraints
        """
        m, basis = prep_SimPleAC()
        basesol = m.localsolve(verbosity=0)

        # Identify signomial constraints
        sp_constraints = find_signomials(m)
        sp_variables = get_varkeys(sp_constraints)
    #
    # def test_SimPleAC_with_treeconstraint(self):
    #     m, basis = prep_SimPleAC()
    #     basesol = m.localsolve(verbosity=0)
    #
    #     # Now replacing the drag model with a learner...
    #     constraints = [c for c in m.flat()]
    #     del constraints[-12:-8]
    #     lnr = iai.read_json("data/airfoil_lnr.json")
    #     for i in range(len(m['C_D'])):
    #         dvar = m['C_D'][i]
    #         ivars = [m['Re'][i], m['\\tau'],
    #                  m['V'][i]/(1.4*287*units('J/kg/K')*250*units('K'))**0.5,
    #                  m['C_L'][i]]
    #         ct = ConstraintTree(lnr, dvar, ivars)
    #         constraints.append(ct)
    #
    #     # Get variable bounds and constraintify those
    #     solutions = pickle.load(open("data/SimPleAC.sol", "rb"))
    #     bounds = get_bounds(solutions)
    #     bounding_constraints = constraints_from_bounds(bounds, m)
    #
    #     gm = GlobalModel(m.cost,
    #                      [constraints, bounding_constraints],
    #                      m.substitutions)
    #     sol = gm.solve(verbosity=0)

    def test_SimPleAC_with_surrogate_tree(self):
        m, basis = prep_SimPleAC()
        # Replicate GP model with new models
        basesol = m.localsolve(verbosity=0)
        dvar = m['W_{f_m}']
        ivars = [m[var] for var in list(basis.keys())]

        # Fitting GPfit model
        solns = pickle.load(open("data/SimPleAC.sol", "rb"))
        subs = pickle.load(open("data/SimPleAC.subs", "rb"))
        X = gen_X(subs, basis)
        Y = [mag(soln['cost'] / basesol['cost']) for soln in solns]
        cstrt, rms = fit(np.log(np.transpose(X)), np.log(Y), 4, 'SMA')

        basis[dvar.key] = basesol('W_{f_m}')
        fit_constraint = constraint_from_gpfit(cstrt, dvar, ivars, basis)
        basis.pop(dvar.key)
        m = Model(dvar, [fit_constraint], basis)
        fitsol = m.solve(verbosity=0, reltol=1e-6)

        # Now with trees
        lnr = iai.read_json("data/SimPleAC_lnr.json")
        basis[dvar.key] =  basesol('W_{f_m}')
        ct = ConstraintTree(lnr, dvar, ivars, basis=basis)
        bounds = pickle.load(open("data/SimPleAC.bounds", "rb"))
        bounding_constraints = constraints_from_bounds(bounds, m)
        gm = GlobalModel(dvar, [bounding_constraints, ct], basis)
        sol = gm.solve(verbosity=0)

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()
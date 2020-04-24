import numpy as np
from gpkit import Variable, Model

import unittest
from gpkit.tests.helpers import run_tests

from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *
from interpretableai import iai
import pickle


from OptimalConstraintTree.constraint_tree import ConstraintTree
from OptimalConstraintTree.global_model import GlobalModel
from OptimalConstraintTree.tools import find_signomials
from OptimalConstraintTree.tools import get_variables, get_bounds

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

    def test_ConstraintTree_sp_constraints(self):
        """
        Tests ConstraintTree generation from SP constraints
        """
        model = Mission(SimPleAC(),4)
        model.cost = model['W_{f_m}']*units('1/N') + model['C_m']*model['t_m']
        basis = {
            'h_{cruise_m}'   :5000*units('m'),
            'Range_m'        :3000*units('km'),
            'W_{p_m}'        :6250*units('N'),
            '\\rho_{p_m}'    :1500*units('kg/m^3'),
            'C_m'            :120*units('1/hr'),
            'V_{min_m}'      :25*units('m/s'),
            'T/O factor_m'   :2,
        }
        model.substitutions.update(basis)
        basesol = model.localsolve(verbosity=0)

        # Identify signomial constraints
        sp_constraints = find_signomials(model)
        sp_variables = get_variables(sp_constraints)

        # Get variable bounds and constraintify those
        solutions = pickle.load(open("data/SimPleAC.sol", "rb"))
        bounds = get_bounds(solutions)

    def test_SimPleAC_with_treeconstraint(self):
        model = Mission(SimPleAC(),4)
        model.cost = model['W_{f_m}']*units('1/N') + model['C_m']*model['t_m']
        basis = {
            'h_{cruise_m}'   :5000*units('m'),
            'Range_m'        :3000*units('km'),
            'W_{p_m}'        :6250*units('N'),
            '\\rho_{p_m}'    :1500*units('kg/m^3'),
            'C_m'            :120*units('1/hr'),
            'V_{min_m}'      :25*units('m/s'),
            'T/O factor_m'   :2,
        }
        model.substitutions.update(basis)
        basesol = model.localsolve(verbosity=0)
        # Now replacing the drag model with a learner...
        constraints = [c for c in model.flat()]
        constraints[-12:-8].clear()
        lnr = iai.read_json("data/airfoil_lnr.json")
        for i in range(len(model['C_D'])):
            dvar = model['C_D'][i]
            ivars = [model['Re'][i], model['\\tau'],
                     model['V'][i]/(1.4*287*units('J/kg/K')*250*units('K'))**0.5,
                     model['C_L'][i]]
            ct = ConstraintTree(lnr, dvar, ivars)
            constraints.append(ct)

        gm = GlobalModel(model.cost, constraints, model.substitutions)
        sol = gm.solve(verbosity=0)

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

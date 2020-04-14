import numpy as np
from gpkit import Variable

import unittest
from gpkit.tests.helpers import run_tests

from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *

from interpretableai import iai
from OptimalConstraintTree.constraint_tree import ConstraintTree

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
                         constraintDict[1].as_hmapslt1({}))

    # def test_ConstraintTree(self):
    #     """
    #     Tests the class ConstraintTree
    #     """
    #     # TODO: figure out why JSON loading is not working...
    #     m = Mission(SimPleAC(),4)
    #     m.cost = m['W_{f_m}']*units('1/N') + m['C_m']*m['t_m']
    #     basis = {
    #         'h_{cruise_m}'   :5000*units('m'),
    #         'Range_m'        :3000*units('km'),
    #         'W_{p_m}'        :6250*units('N'),
    #         '\\rho_{p_m}'    :1500*units('kg/m^3'),
    #         'C_m'            :120*units('1/hr'),
    #         'V_{min_m}'      :25*units('m/s'),
    #         'T/O factor_m'   :2,
    #     }
    #     m.substitutions.update(basis)
    #     basesol = m.localsolve(verbosity=0)
    #     # Adding fits for each mission segment
    #     # Note that Mach number does not exist for the original model,
    #     # so it has to be added as a monomial.
    #     cts = []
    #     for i in range(len(m['C_D'])):
    #         dvar = m['C_D'][i]
    #         ivars = [m['Re'][i], m['\\tau'],
    #                  m['V'][i]/(1.4*287*units('J/kg/K')*250*units('K'))**0.5,
    #                  m['C_L'][i]]
    #         cts.append(ConstraintTree(lnr, dvar, ivars))

    # def test_ConstraintTree_in_model(self):
    #     """
    #     Tests the class ConstraintTree
    #     """
    #     # TODO: figure out why JSON loading is not working...
    #     m = Mission(SimPleAC(),4)
    #     m.cost = m['W_{f_m}']*units('1/N') + m['C_m']*m['t_m']
    #     basis = {
    #         'h_{cruise_m}'   :5000*units('m'),
    #         'Range_m'        :3000*units('km'),
    #         'W_{p_m}'        :6250*units('N'),
    #         '\\rho_{p_m}'    :1500*units('kg/m^3'),
    #         'C_m'            :120*units('1/hr'),
    #         'V_{min_m}'      :25*units('m/s'),
    #         'T/O factor_m'   :2,
    #     }
    #     m.substitutions.update(basis)
    #     basesol = m.localsolve(verbosity=0)
    #     # Adding fits for each mission segment
    #     # Note that Mach number does not exist for the original model,
    #     # so it has to be added as a monomial.
    #     cts = []
    #     for i in len(m['C_D']):
    #         ivar = m['C_D'][i]
    #         ivars = [m['Re'][i], m['\\tau'],
    #                  m['V'][i]/(1.4*287*units('J/kg/K')*250*units('K'))**0.5,
    #                  m['C_L'][i]]
    #         cts.append(ConstraintTree(lnr, ivar, ivars))

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

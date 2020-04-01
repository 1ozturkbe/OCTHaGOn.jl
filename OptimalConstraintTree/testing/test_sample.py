""" Tests for function sampling methods. """
import numpy as np
from numpy.random import exponential as xp
from gpkit import Variable
import unittest
from gpkit.tests.helpers import run_tests
import pickle
from gpkitmodels.SP.SimPleAC.SimPleAC_mission import *

from OptimalConstraintTree.sample import sample_gpobj, sample_gpmodel


class TestSample(unittest.TestCase):
    """ Tests sampling methods. """
    def test_sample_gpobj(self):
        # TODO: compare different sampling methods here.
        a = Variable()
        b = Variable()
        gpobjs = [a*b**(-1.2), a + b**(2.1)]
        samples = 10
        for gpobj in gpobjs:
            bounds = {a.key: [xp(), xp()], b.key: [xp(), xp()]}
            Yvals, Xsubs = sample_gpobj(gpobj, bounds, samples)

    def test_sample_gpmodel(self):
        m = Mission(SimPleAC(),4)
        m.cost = m['W_{f_m}']*units('1/N') + m['C_m']*m['t_m']
        subs = {
            'h_{cruise_m}'   :5000*units('m'),
            'Range_m'        :3000*units('km'),
            'W_{p_m}'        :6250*units('N'),
            '\\rho_{p_m}'    :1500*units('kg/m^3'),
            'C_m'            :120*units('1/hr'),
            'V_{min_m}'      :25*units('m/s'),
            'T/O factor_m'   :2,
        }
        bounds = {
            'h_{cruise_m}'   :[6000*units('m'), 1000*units('m')], # reversed
            'Range_m'        :[1000*units('km'), 5000*units('km')],
            'W_{p_m}'        :[1250*units('N'), 6250*units('N')],
            '\\rho_{p_m}'    :[1000*units('kg/m^3'), 2000*units('kg/m^3')],
            'C_m'            :[100*units('1/hr'), 200*units('1/hr')],
            'V_{min_m}'      :[15*units('m/s'), 45*units('m/s')],
            'T/O factor_m'   :[2, 3],
        }
        samples = 300
        solns, subs = sample_gpmodel(m, bounds, samples, verbosity=-1)
        pickle.dump(bounds, open("subs/SimPleAC.bounds", "wb"))
        for i in range(samples):
            solns[i].save("solns/SimPLeAC" + str(i) + ".sol")
            pickle.dump(subs[i], open("subs/SimPLeAC" + str(i) + ".subs", "wb"))

TESTS = [TestSample]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

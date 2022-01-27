""" Tests for function sampling methods. """
import numpy as np
from numpy.random import exponential as xp
from gpkit import Variable, units
import unittest
from OCTHaGOn.testing.run_tests import run_tests
import pickle

from OCTHaGOn.testing.test_ort_models import prep_SimPleAC
from OCTHaGOn.sample import sample_gpobj, sample_gpmodel
from OCTHaGOn.sample import gen_X

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
            _, _ = sample_gpobj(gpobj, bounds, samples)

    def test_sample_gpmodel(self):
        m, basis = prep_SimPleAC()
        bounds = {
            m['h_{cruise_m}'].key   :[6000*units('m'), 1000*units('m')], # reversed
            m['Range_m'].key        :[1000*units('km'), 5000*units('km')],
            m['W_{p_m}'].key        :[1250*units('N'), 6250*units('N')],
            m['\\rho_{p_m}'].key    :[1000*units('kg/m^3'), 2000*units('kg/m^3')],
            m['C_m'].key            :[100*units('1/hr'), 200*units('1/hr')],
            m['V_{min_m}'].key      :[15*units('m/s'), 45*units('m/s')],
            m['T/O factor_m'].key   :[2, 3],
        }
        samples = 500
        solns, subs = sample_gpmodel(m, bounds, samples, verbosity=-1, filename="data/SimPleAC")

    def test_gen_X(self):
        a = Variable()
        _, basis = prep_SimPleAC()
        basis.update({a: 3})
        subs = pickle.load(open("data/SimPleAC.subs", "rb"))
        with self.assertRaises(AttributeError):
            gen_X(subs, basis)
        basis.pop(a)
        basis.update({a.key: 3})
        with self.assertRaises(KeyError):
            gen_X(subs, basis)
        basis.pop(a.key)
        _ = gen_X(subs, basis)

TESTS = [TestSample]

if __name__ == "__main__":
    run_tests(TESTS)
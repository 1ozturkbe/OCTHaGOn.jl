""" Tests for function sampling methods. """
import numpy as np
from gpkit import Variable
import unittest
from gpkit.tests.helpers import run_tests

from OptimalConstraintTree.sample import lh_sample_gpobj

class TestSample(unittest.TestCase):
    """ Tests sampling methods. """
    def test_lh_sample_gpobj(self):
        # TODO: compare different sampling methods here.
        a = Variable()
        b = Variable()
        xp = np.random.exponential
        gpobjs = [a*b**(-1.2), a + b**(2.1)]
        samples = 10
        for gpobj in gpobjs:
            bounds = {a.key: [xp(), xp()], b.key: [xp(), xp()]}
            _, _ = lh_sample_gpobj(gpobj, bounds, samples)

    def test_lh_sample_gpmodel(self):
        # TODO: Sample GP model here.
        pass

TESTS = [TestSample]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

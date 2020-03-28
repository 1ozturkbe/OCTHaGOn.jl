import numpy as np
from gpkit import Variable

import unittest
from gpkit.tests.helpers import run_tests

from OptimalConstraintTree.constraintify import monomials_from_pwl_data

class TestConstraintify(unittest.TestCase):
    def test_monomials_from_pwl_data(self):
        pwlDict = {1: [1, [2,3,4]],
                   2: [5, [6,7,8]],
                   3: [9, [10,11,12]]}
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        gpvars = [a, b, c]
        monys = monomials_from_pwl_data(pwlDict, gpvars)
        self.assertEqual(monys[1], np.exp(1)*a**2*b**3*c**4)

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

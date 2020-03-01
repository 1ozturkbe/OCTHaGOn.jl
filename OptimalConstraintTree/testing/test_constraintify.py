from builtins import zip
from builtins import range
import numpy as np
from gpkit import Variable, Model
import scipy.optimize as op
import os

import unittest
from gpkit.tests.helpers import run_tests

np.warnings.filterwarnings('ignore')

class TestConstraintify(unittest.TestCase):
    def test_gp_constraintify(self):
        pwlConstraintData = [[1, [2,3,4]],
                             [5, [6,7,8]],
                             [9, [10,11,12]]]
        return

TESTS = [TestConstraintify]

def test():
    run_tests(TESTS)

if __name__ == "__main__":
    test()

import numpy as np
from gpkit import Variable, Model, VectorVariable, SignomialsEnabled

import unittest
from OCTHaGOn.testing.run_tests import run_tests

class TestExamples(unittest.TestCase):
    def test_example1(self):
        obj = Variable('objective')
        x = VectorVariable(3, 'x')
        lbs = [70, 1.00001, 0.5]
        ubs = [150, 20, 21]
        constraints = [100 >= x[1]/x[2] + x[1] + 0.05*x[0]*x[2]]
        for i in range(3):
            constraints.extend([x[i] >= lbs[i],
                                x[i] <= ubs[i]])
        with SignomialsEnabled():
            constraints.extend([obj <= x[0] + 5/x[1] - 0.5*x[0]/x[1]])
        m = Model(1/obj, constraints)
        sol = m.localsolve(verbosity=4)

TESTS = [TestExamples]

if __name__ == "__main__":
    run_tests(TESTS)
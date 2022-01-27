"""Script for running all unit tests"""
import gpkit
from gpkit.tests.run_tests import run
from gpkit.tests.helpers import run_tests as r_t
from gpkit.tests.test_repo import git_clone, pip_install

import julia.core

def import_tests():
    """Get a list of all robust unit test TestCases"""
    tests = []

    from OCTHaGOn.testing import test_constraintify
    tests += test_constraintify.TESTS

    from OCTHaGOn.testing import test_ort_models
    tests += test_ort_models.TESTS

    from OCTHaGOn.testing import test_sample
    tests += test_sample.TESTS

    return tests

def run_tests(tests, xmloutput=None, verbosity=2):
    if julia.core.is_windows:
        r_t(tests, xmloutput=xmloutput, verbosity=verbosity)
    else:
        from julia.api import Julia
        Julia(compiled_modules=False)
        r_t(tests, xmloutput=xmloutput, verbosity=verbosity)

def test():
    try:
        import gpkitmodels
    except:
        git_clone("gplibrary")
        pip_install("gplibrary", local=True)
    alltests = import_tests()
    TESTS = []
    for testcase in alltests:
        for solver in gpkit.settings["installed_solvers"]:
            if solver:
                test = type(str(testcase.__name__+"_"+solver),
                            (testcase,), {})
                setattr(test, "solver", solver)
                TESTS.append(test)
    run(tests=TESTS, xmloutput=True)

if __name__ == '__main__':
    if julia.core.is_windows:
        test()
    else:
        from julia.api import Julia
        Julia(compiled_modules=False)
        test()
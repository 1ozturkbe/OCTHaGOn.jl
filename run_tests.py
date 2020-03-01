"""Script for running all unit tests"""
import gpkit
from gpkit.tests.run_tests import run
from gpkit.tests.test_repo import git_clone, pip_install

def import_tests():
    """Get a list of all robust unit test TestCases"""
    tests = []

    from OptimalConstraintTree.testing import test_constraintify
    tests += test_constraintify.TESTS

    return tests

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
    test()

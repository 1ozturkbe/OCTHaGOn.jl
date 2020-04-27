import numpy as np
from time import time

from gpkit import Model
from gpkit.exceptions import (InvalidGPConstraint, Infeasible,
                              InvalidSGPConstraint)
from gpkit.small_scripts import mag

from OptimalConstraintTree.constraint_tree import ConstraintTree
from OptimalConstraintTree.tools import flatten
from gpkit import ConstraintSet
from gpkit.keydict import KeySet

EPS = 1e-6

class GlobalModel(Model):
    """ Extends the GPkit Model class to be able to accommodate
    constraint trees.

    Arguments
    ---------
    cost : Posynomial

    constraints : ConstraintSet or list of constraints or ConstraintTrees.

    substitutions : dict
        Note: Do not forget substitutions required for ConstraintTrees and
        GP object fits.
    """
    sps = None
    gp_vars = None
    sp_vars = None
    tree_vars = None

    def __init__(self, cost, constraints, *args, **kwargs):
        self.cost = cost
        self.classify_constraints(constraints)
        self.substitutions = kwargs.pop("substitutions", None)
        for key, value in kwargs.items():
            if key == 'solve_type':
                for tree in self.constraints['trees']:
                    tree.solve_type = value
                    tree.setup()
        self.sp_model = Model(cost, [self.constraints['gp_constraints'],
                                     self.constraints['sp_constraints']],
                              self.substitutions, *args, **kwargs)

    def classify_constraints(self, constraints):
        self.constraints = {'trees': [],
                       'sp_constraints': [],
                       'gp_constraints': []}
        self.gp_vars, self.sp_vars, self.tree_vars = set(), set(), set()
        for constr in flatten(constraints):
            if isinstance(constr, ConstraintTree):
                self.tree_vars.update(constr.varkeys)
                self.constraints['trees'].append(constr)
            elif hasattr(constr, 'as_gpconstr'):
                self.sp_vars.update(constr.varkeys)
                self.constraints['sp_constraints'].append(constr)
            elif hasattr(constr, 'as_hmapslt1'):
                self.gp_vars.update(constr.varkeys)
                self.constraints['gp_constraints'].append(constr)
            else:
                raise Warning("Constraint %s could not be "
                              "classified." % constr)
        return self.constraints

    def solve(self, solver=None, verbosity=0, reltol=1e-3, x0=None, iteration_limit=50):
        """
        Solves GlobalModel using defined solve_type (currently
        only works as a sequential SP solver).
        Modeled after GPkit .localsolve method.
        :param verbosity:
        :param reltol:
        :param x0: initial guess
        :return: SolutionArray
        """
        self.sps, self.solver_outs, self._results = [], [], []
        starttime = time()
        if verbosity > 0:
            print("Starting a sequence of GlobalModel solves...")
            # print(" for %i free variables" % len(self.sgpvks))
            # print("  in %i locally-GP constraints" % len(self.sgpconstraints))
            # print("  and for %i free variables" % len(self._gp.varlocs))
            # print("       in %i posynomial inequalities." % len(self._gp.k))
        if x0:
            xi = x0.copy()
        else:
            if verbosity >= 2:
                print("Generating initial first guess using provided SP"
                      "constraints and bounds.")
            xi = self.sp_model.debug(verbosity=0)

        # Starting solve...
        prevcost, cost, rel_improvement = None, None, None
        base_constraints = self.constraints['gp_constraints']
        base_constraints.append(self.constraints['sp_constraints'])
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.sps) > iteration_limit:
                raise Infeasible(
                    "Unsolved after %s iterations. Check `m.sps` to check"
                    " if solutions they're converging." % len(self.sps))
            tree_constraints = [tree.get_leaf_constraints(xi) for tree in self.constraints['trees']]
            self.sps.append(Model(self.cost, [base_constraints, tree_constraints],
                                  self.substitutions))
            if verbosity > 1:
                print("\nSP Solve %i" % len(self.sps))
            if verbosity > 2:
                print("===============")
            try:
                solver_out = self.sps[-1].solve(solver, verbosity=verbosity,
                                                reltol=reltol, x0=x0)
            except InvalidGPConstraint:
                solver_out = self.sps[-1].localsolve(solver, verbosity=verbosity,
                                                     reltol=reltol, x0=x0)
            self.solver_outs.append(solver_out)
            cost = float(solver_out["objective"])
            x0 = solver_out
            if verbosity > 2:
                result = self.sps.program.gps[-1].generate_result(solver_out, verbosity=verbosity-3)
                self._results.append(result)
                print(result.table())
            elif verbosity > 1:
                print("Solved cost was %.4g." % cost)
            if prevcost is None:
                continue
            rel_improvement = (prevcost - cost) / (prevcost + cost)
            if cost * (1 - EPS) > prevcost + EPS and verbosity > -1:
                print("SSP not convergent: Cost rose by %.2g%% on SP solve %i."
                      " Details can be found in `m.sps.results` or by"
                      " solving at a higher verbosity. Note that convergence is"
                      " not guaranteed for GlobalModels.\n"
                      % (100 * (cost - prevcost) / prevcost, len(self.sps)))
                rel_improvement = cost = None
        self.result = self.sps[-1].program.gps[-1].generate_result(solver_out, verbosity=verbosity-3)
        self.result["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print("Solving took %.3g seconds and %i SP solves."
                  % (self.result["soltime"], len(self.sps)))
        #TODO: add slack checking as well.
        return xi


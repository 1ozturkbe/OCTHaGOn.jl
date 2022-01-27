import numpy as np
from time import time
from copy import copy

from gpkit import Model
from gpkit.exceptions import (InvalidGPConstraint, Infeasible,
                              InvalidSGPConstraint)
from gpkit.small_scripts import mag

from OCTHaGOn.constraint_tree import ConstraintTree
from OCTHaGOn.tools import flatten, HiddenPrints, get_active_bounds

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
    substitutions = None

    def __init__(self, cost, constraints, *args, **kwargs):
        self.cost = cost
        self.classify_constraints(constraints)
        self.substitutions = kwargs.pop("substitutions", None)
        if args and not self.substitutions:
            self.substitutions, = args
        for key, value in kwargs.items():
            if key == 'solve_type':
                for tree in self.constraints['trees']:
                    tree.solve_type = value
                    tree.setup()

    def classify_constraints(self, constraints):
        self.constraints = {'trees': [],
                            'bound_constraints': [],
                            'sp_constraints': [],
                            'gp_constraints': []}
        self.gp_vars, self.sp_vars, self.tree_vars = set(), set(), set()
        for constr in flatten(constraints):
            if isinstance(constr, ConstraintTree):
                self.tree_vars.update(constr.varkeys)
                self.constraints['trees'].append(constr)
                self.constraints['bound_constraints'].append(constr.bound_constraints)
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
            sp_model = Model(self.cost, [self.constraints['gp_constraints'],
                                         self.constraints['sp_constraints'],
                                         self.constraints['bound_constraints']],
                                      substitutions=copy(self.substitutions))
            self.sps.append(sp_model)
            if verbosity > 1:
                print("Generating initial first guess using provided "
                      " constraints and bounds.")
                print("\n[Debug] Solve %i" % len(self.sps))
            with HiddenPrints():
                xi = self.sps[-1].debug(verbosity=verbosity-2)
            # TODO: figure out how to properly update results.
            # if verbosity > 2:
                # try:
                #     # self.solver_outs.append(self.sps[-1].program.solver_out)
                #     result = self.sps[-1].program.generate_result(self.solver_outs[-1], verbosity=verbosity-3)
                # except (AttributeError, InvalidGPConstraint):
                #     # self.solver_outs.append(self.sps[-1].program.gps[-1].solver_out)
                #     result = self.sps[-1].program.gps[-1].generate_result(self.solver_outs[-1], verbosity=verbosity-3)
                # self._results.append(result)
        # Starting solve...
        prevcost, cost, rel_improvement = None, None, None
        base_constraints = self.constraints['gp_constraints']
        base_constraints.append(self.constraints['sp_constraints'])
        base_constraints.append(self.constraints['bound_constraints'])
        while rel_improvement is None or rel_improvement > reltol:
            prevcost = cost
            if len(self.sps) > iteration_limit:
                raise Infeasible(
                    "Unsolved after %s iterations. Check `m.sps` to check"
                    " if solutions they're converging." % len(self.sps))
            tree_constraints = [tree.get_leaf_constraints(xi) for tree in self.constraints['trees']]
            self.sps.append(Model(self.cost, [base_constraints, tree_constraints],
                                  copy(self.substitutions)))
            try:
                if verbosity > 1:
                    print("\n[GP] Solve %i" % len(self.sps))
                xi = self.sps[-1].solve(solver, verbosity=verbosity-2,
                                                reltol=reltol, x0=xi)
                self.solver_outs.append(self.sps[-1].program.solver_out)
            except InvalidGPConstraint:
                if verbosity > 1:
                    print("\n[SP] Solve %i" % len(self.sps))
                xi = self.sps[-1].localsolve(solver, verbosity=verbosity-2,
                                                     reltol=reltol, x0=xi)
                self.solver_outs.append(self.sps[-1].program.gps[-1].solver_out)
            if verbosity > 2:
                print("===============")
            cost = float(xi["cost"])
            if verbosity > 2:
                try:
                    result = self.sps[-1].program.generate_result(self.solver_outs[-1], verbosity=verbosity-3)
                except AttributeError:
                    result = self.sps[-1].program.gps[-1].generate_result(self.solver_outs[-1], verbosity=verbosity-3)
                self._results.append(result)
            elif verbosity > 1:
                print("Solved cost was %.4g." % cost)
            if prevcost is None:
                continue
            rel_improvement = (prevcost - cost) / (prevcost + cost)
            if cost * (1 - EPS) > prevcost + EPS and verbosity > -1:
                print("SSP not convergent: Cost rose by %.2g%% on solve %i."
                      " Details can be found in `m.sps.results` or by"
                      " solving at a higher verbosity. Note that convergence is"
                      " not guaranteed for GlobalModels.\n"
                      % (100 * (cost - prevcost) / prevcost, len(self.sps)))
                rel_improvement = cost = None
        try:
            self.result = self.sps[-1].program.generate_result(self.solver_outs[-1], verbosity=verbosity-3)
        except AttributeError:
            self.result = self.sps[-1].program.gps[-1].generate_result(self.solver_outs[-1], verbosity=verbosity-3)
        self.result["soltime"] = time() - starttime
        if verbosity > 1:
            print()
        if verbosity > 0:
            print("Solving took %.3g seconds and %i SP solves."
                  % (self.result["soltime"], len(self.sps)))
            if self.constraints['bound_constraints']:
                self.active_bounds = get_active_bounds(self, xi)
                if self.active_bounds:
                    print("\nSolves with these monomials hitting bounds "
                          "(from highest to lowest sensitivity):")
                    for item in self.active_bounds:
                        print("\n %s is %s bounded at %s, with sens. %s" % item)
        # TODO: add slack checking as well
        return xi


import numpy as np
from time import time

from gpkit import Model
from gpkit.exceptions import InvalidGPConstraint
from gpkit.small_scripts import mag

from OptimalConstraintTree.constraint_tree import ConstraintTree
from gpkit import ConstraintSet

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

    def __init__(self, cost, constraints, *args, **kwargs):
        self.cost = cost
        self.trees = [c for c in constraints if isinstance(c, ConstraintTree)]
        self.sp_constraints = ConstraintSet([c for c in constraints if not isinstance(c, ConstraintTree)])
        # self.treevars = set([[tree.dvar, ivar for ivar in tree.ivars]
        #                     for tree in self.trees])
        for key, value in kwargs.items():
            if key == 'solve_type':
                for tree in self.trees:
                    tree.solve_type = value
                tree.setup()
        print(cost)
        self.sp_model = Model(cost, self.sp_constraints, *args, **kwargs)

    def solve(self, verbosity=0, reltol=1e-3, x0=None):
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
            print("Starting a sequence of GlobalModel solves")
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
        self.sps = []

        # Starting solve...
        prev_cost = np.inf
        new_cost = 1e30
        # prevcost, cost, rel_improvement = None, None, None
        # while rel_improvement is None or rel_improvement > reltol:
        #     prevcost = cost
        #     if len(self.gps) > iteration_limit:
        #         raise Infeasible(
        #             "Unsolved after %s iterations. Check `m.program.results`;"
        #             " if they're converging, try `.localsolve(...,"
        #             " iteration_limit=NEWLIMIT)`." % len(self.gps))
        while prev_cost/new_cost - 1 >= reltol:
            constraints = self.sp_constraints.copy()
            for tree in self.trees:
                constraints.extend(tree.get_leaf_constraints(xi))
            self.sps.append(Model(self.cost, constraints, self.sp_model.substitutions))
            try:
                xi = self.sps[-1].solve(verbosity=verbosity)
            except InvalidGPConstraint:
                xi = self.sps[-1].localsolve(verbosity=verbosity)
            prev_cost = new_cost
            new_cost = mag(xi['cost'])
        return xi


        #     gp = self.gp(x0, cleanx0=True)
        #     self.gps.append(gp)  # NOTE: SIDE EFFECTS
        #     if verbosity > 1:
        #         print("\nGP Solve %i" % len(self.gps))
        #     if verbosity > 2:
        #         print("===============")
        #     solver_out = gp.solve(solver, verbosity=verbosity-1,
        #                           gen_result=False, **solveargs)
        #     self.solver_outs.append(solver_out)
        #     cost = float(solver_out["objective"])
        #     x0 = dict(zip(gp.varlocs, np.exp(solver_out["primal"])))
        #     if verbosity > 2:
        #         result = gp.generate_result(solver_out, verbosity=verbosity-3)
        #         self._results.append(result)
        #         print(result.table(self.sgpvks))
        #     elif verbosity > 1:
        #         print("Solved cost was %.4g." % cost)
        #     if prevcost is None:
        #         continue
        #     rel_improvement = (prevcost - cost)/(prevcost + cost)
        #     if cost*(1 - EPS) > prevcost + EPS and verbosity > -1:
        #         print("SGP not convergent: Cost rose by %.2g%% on GP solve %i."
        #               " Details can be found in `m.program.results` or by"
        #               " solving at a higher verbosity. Note that convergence is"
        #               " not guaranteed for models with SignomialEqualities.\n"
        #               % (100*(cost - prevcost)/prevcost, len(self.gps)))
        #         rel_improvement = cost = None
        # # solved successfully!
        # self.result = gp.generate_result(solver_out, verbosity=verbosity-3)
        # self.result["soltime"] = time() - starttime
        # if verbosity > 1:
        #     print()
        # if verbosity > 0:
        #     print("Solving took %.3g seconds and %i GP solves."
        #           % (self.result["soltime"], len(self.gps)))
        # if hasattr(self.slack, "key"):
        #     excess_slack = self.result["variables"][self.slack.key] - 1  # pylint: disable=no-member
        #     if excess_slack <= EPS:
        #         del self.result["freevariables"][self.slack.key]  # pylint: disable=no-member
        #         del self.result["variables"][self.slack.key]  # pylint: disable=no-member
        #         del self.result["sensitivities"]["variables"][self.slack.key]  # pylint: disable=no-member
        #         slackconstraint = self.gpconstraints[0]
        #         del self.result["sensitivities"]["constraints"][slackconstraint]
        #     elif verbosity > -1:
        #         print("Final solution let signomial constraints slacken by"
        #               " %.2g%%. Calling .localsolve with a higher"
        #               " `pccp_penalty` (it was %.3g this time) will reduce"
        #               " final slack if the model is solvable with less. If"
        #               " you think it might not be, check by solving with "
        #               "`use_pccp=False, x0=(this model's final solution)`.\n"
        #               % (100*excess_slack, self.pccp_penalty))
        # return self.result
"""
Defines the ConstraintTree class, with class and static methods.
"""

from gpkit.small_scripts import mag

import numpy as np
from interpretableai import iai

class ConstraintTree:
    """ ConstraintTree class, containing the approximate constraints
    from ORT models"""

    pwl_data = None
    tr_data = None
    tr_constraints = None
    pwl_constraints = None

    def __init__(self, lnr, dvar, ivars, **kwargs):
        self.learner = lnr  # original PWL learner
        kwarg_keys = set(['basis', 'solve_type', 'epsilon'])
        self.__dict__.update((key, None) for key in kwarg_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items()
                             if key in kwarg_keys)
        if self.basis:
            self.dvar = dvar / self.basis[dvar.varkey]
            self.ivars = [dvar / self.basis[ivar.varkey] for ivar in ivars]
        else:
            self.dvar = dvar
            self.ivars = ivars
        for var in set([self.dvar, self.ivars]):
            if var.units:
                raise ValueError('Monomial %s has units %s but'
                                 'is required to be unitless. '
                                 % (var, var.units))
        if not self.solve_type:
            self.solve_type = 'seq'  # Default sequential solver
        self.constraints = {}

    def setup(self):
        # Generating constraint data from ORT
        self.pwl_data = self.pwl_constraint_data(self.learner)
        self.tr_data = self.trust_region_data(self.learner)
        if self.solve_type == "seq":
            self.tr_constraints = self.tr_constraintify(self.tr_data[0],
                                                        self.tr_data[1],
                                                        self.ivars,
                                                        self.epsilon)
            self.pwl_constraints = self.pwl_constraintify(self.pwl_data,
                                                          self.dvar,
                                                          self.ivars,
                                                          self.epsilon)
        elif self.solve_type == "mi":
            raise Warning("Mixed integer solver not yet available.")
        elif self.solve_type == "global":
            raise Warning("Global solver not yet available.")
        else:
            raise ValueError("ConstraintTree with solver type %s "
                             "is not supported." % self.solve_type)

    def get_leaf_constraint(self, sol):
        """ Returns constraints for a particular leaf
        depending on the features (solution of free vars). """
        # TODO: finish
        inps = np.array([np.log(mag(sol))])
        leaf_no = self.learner.apply(inps)
        return self.constraints[leaf_no]

    def piecewise_convexify(self, rms_threshold=0.01, max_threshold=0.02):
        """ Prunes the tree using by efficiently computing adjacent convex regions,
        returning a new tree. """
        pass

    @staticmethod
    def pwl_constraint_data(lnr: iai.OptimalTreeRegressor, vks=None):
        """
        Creates PWL dataset from a OptimalTreeLearner
        Arguments:
            lnr: OptimalTreeLearner
            vks: varkeys, ['x1',..., 'xp'] for numpy arrays
                 potentially different for other data structures
        Returns:
            Dict[leaf_number] containing [B0 (offset), B (linear)]
        """
        n_nodes = lnr.get_num_nodes()
        if not vks:
            vks = ['x' + str(i) for i in range(1, len(lnr.variable_importance()) + 1)]
        all_leaves = [i for i in range(1, n_nodes + 1) if lnr.is_leaf(i)]  # Julia is one-indexed!
        pwlDict = {leaf: [] for leaf in all_leaves}
        for i in range(len(all_leaves)):
            β0 = lnr.get_regression_constant(all_leaves[i])
            weights = lnr.get_regression_weights(all_leaves[i])[0]
            β = []
            for j in range(len(vks)):
                if vks[j] in weights.keys():
                    β.append(weights[vks[j]])
                else:
                    β.append(0.)
            pwlDict[all_leaves[i]] = [β0, β]
        return pwlDict

    @staticmethod
    def trust_region_data(lnr: iai.OptimalTreeRegressor, vks=None):
        """
        Creates trust region from a OptimalTreeLearner
        Arguments:
            lnr: OptimalTreeLearner
            vks: varkeys, ['x1',..., 'xp'] for numpy arrays
                 potentially different for other data structures
        Returns:
            upper and lowerDict, with [leaf_number] containing [threshold, coeffs]
        """
        n_nodes = lnr.get_num_nodes()
        if not vks:
            vks = ['x' + str(i) for i in range(1, len(lnr.variable_importance()) + 1)]
        all_leaves = [i for i in range(1, n_nodes + 1) if lnr.is_leaf(i)]  # Julia is one-indexed!
        upperDict = {leaf: [] for leaf in all_leaves}
        lowerDict = {leaf: [] for leaf in all_leaves}
        for i in range(len(all_leaves)):
            # Find all parents
            parents = [all_leaves[i]]
            while lnr.get_depth(parents[-1]) > 0:
                parents.append(lnr.get_parent(parents[-1]))
            for j in parents[1:]:
                # For each parent, define trust region with binary variables
                threshold = lnr.get_split_threshold(j)
                if lnr.is_hyperplane_split(j):
                    weights = lnr.get_split_weights(j)[0]
                else:
                    feature = lnr.get_split_feature(j)
                    weights = {feature: 1}
                upper = lnr.get_upper_child(j) in parents  # Checking upper vs. lower split
                α = []
                for k in range(len(vks)):
                    if vks[k] in weights.keys():
                        α.append(weights[vks[k]])
                    else:
                        α.append(0.)
                if upper:
                    upperDict[all_leaves[i]].append([threshold, α])
                else:
                    lowerDict[all_leaves[i]].append([threshold, α])
        return upperDict, lowerDict

    @staticmethod
    def tr_constraintify(upperDict, lowerDict, ivars, epsilon=None):
        """
        Turns trust region data into monomial inequality constraints.
        :param upperDict: Dict of upper split data
        :param lowerDict: Dict of lower split data
        :param dvar: dependent monomials
        :param ivars: independent monomials
        :param epsilon: whether or not to add any trust region inflation
        :return: dict of monomial inequality constraints by leaf
        """
        if list(upperDict.keys()) != list(lowerDict.keys()):
            raise ValueError("There is an issue with the keys in "
                             "trust region constraint generation.")
        constraintDict = {key: [] for key in list(upperDict.keys())}
        if epsilon:
            epsilon = epsilon
        else:
            epsilon = 1.
        for leaf_key in list(constraintDict.keys()):
            for constraint_data in upperDict[leaf_key]:
                bound = float(np.exp(constraint_data[0]))
                mono = np.prod([ivars[j] ** constraint_data[1][j]
                                for j in range(len(constraint_data[1]))]).value
                constraintDict[leaf_key].append(bound <= mono ** epsilon)
            for constraint_data in lowerDict[leaf_key]:
                bound = float(np.exp(constraint_data[0]))
                mono = np.prod([ivars[j] ** constraint_data[1][j]
                                for j in range(len(constraint_data[1]))]).value
                constraintDict[leaf_key].append(bound ** epsilon >= mono)
        return constraintDict

    @staticmethod
    def pwl_constraintify(pwlDict, dvar, ivars, epsilon=None):
        """
        Turns pwl constraint data into monomial inequality constraints
        Arguments:
            pwlDict: Dict[leaf_number] containing [B0 (offset), B (linear)]
            dvar: dependent monomial
            ivars: independent monomials
            epsilon: optional offset variable
        Returns:
            Dict[leaf_number] containing monomial
        """
        constraintDict = {key: [] for key in list(pwlDict.keys())}
        for key, value in pwlDict.items():
            c = np.exp(value[0])
            if len(ivars) != len(value[1]):
                raise ValueError("Number of GP variables don't match "
                                 "approximation dimension.")
            mono = np.prod([ivars[j] ** value[1][j]
                            for j in range(len(value[1]))]).value
            constraintDict[key] = c * mono <= dvar
        return constraintDict

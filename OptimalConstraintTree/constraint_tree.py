"""
Defines the ConstraintTree class, with class and static methods.
"""

from gpkit.small_scripts import mag
from gpkit import Variable, VectorVariable, SignomialsEnabled
from gpkit import SignomialEquality
from gpkit.nomials import PosynomialInequality
from gpkit.keydict import KeySet

from OptimalConstraintTree.tools import mergeDict, check_units

import numpy as np
from interpretableai import iai

from pandas.core.frame import DataFrame as df

class ConstraintTree:
    """ ConstraintTree class, containing the approximate constraints
    from ORT models. Also includes many static models for tree
    data and constraint generation. """
    # pylint: disable=too-many-instance-attributes
    basis = None
    bounds = None
    solve_type = 'seq'
    epsilon = 0
    oper = '>='
    pwl_data = None
    tr_data = None
    tr_constraints = None
    pwl_constraints = None
    M = 10
    _varkeys = None

    def __init__(self, lnr, dvar, ivars, **kwargs):
        """
        Init's a ConstraintTree.
        :param lnr: OptimalRegressionTreeLearner
        :param dvar: GPkit dependent monomial
        :param ivars: GPkit independent monomial
        :param kwargs: basis required if dvar and ivars are not dimensionless
        """
        self.learner = lnr  # original PWL learner
        self.norm_dvar = None
        self.norm_ivars = []
        self.dvar = dvar
        self.ivars = ivars
        self.__dict__.update((key, value) for key, value in kwargs.items())
        if self.basis:
            if dvar.key in self.basis.keys():
                self.norm_dvar = dvar / self.basis[dvar.key]
            else:
                self.norm_dvar = dvar
            self.norm_ivars = []
            for ivar in self.ivars:
                if ivar.key in self.basis.keys():
                    self.norm_ivars.append(ivar / self.basis[ivar.key])
                else:
                    self.norm_ivars.append(ivar)
        else:
            self.norm_dvar = dvar
            self.norm_ivars = ivars
        check_units(self.norm_dvar)
        check_units(self.norm_ivars)
        if not self.solve_type:
            self.solve_type = 'seq'  # Default sequential solver
        try:
            self.constraintify()
        except OverflowError:
            print("The learner %s \n has regression coefficients "
                  "that are too large. "
                  "Please increase the regression_lambda and "
                  "try again. " % self.learner)

    def constraintify(self):
        """
        Generates all relevant PWL and trust region data.
        """
        self.pwl_data = self.pwl_constraint_data(self.learner)
        self.tr_data = self.trust_region_data(self.learner)
        if self.solve_type == "seq":
            self.constraints = mergeDict(self.tr_constraintify(self.tr_data[0],
                                                        self.tr_data[1],
                                                        self.norm_ivars,
                                                        self.epsilon),
                                         self.pwl_constraintify(self.pwl_data,
                                                          self.norm_dvar,
                                                          self.norm_ivars,
                                                          self.oper))
        elif self.solve_type == "mi":
            self.constraints = self.bigM_constraintify(self.tr_data[0],
                                                       self.tr_data[1],
                                                       self.pwl_data,
                                                       self.norm_dvar,
                                                       self.norm_ivars,
                                                       oper=self.oper,
                                                       M=self.M)
        elif self.solve_type == "global":
            raise Warning("Global solver not yet available.")
        else:
            raise ValueError("ConstraintTree with solver type %s "
                             "is not supported." % self.solve_type)
    @property
    def varkeys(self):
        if self._varkeys is None:
            vks = [var.key for var in self.ivars]
            vks.append(self.dvar.key)
            self._varkeys = KeySet(vks)
        return self._varkeys

    def get_leaf_constraints(self, sol):
        """ Returns constraints for a particular leaf
        depending on the features (solution of free vars). """
        inps = df(np.array([np.log(mag(sol(var))) for var in self.norm_ivars]))
        inps = inps.transpose()
        leaf_no = self.learner.apply(inps.values)
        return self.constraints[leaf_no[0]]

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
    def tr_constraintify(upperDict, lowerDict, ivars, epsilon=0):
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
        for leaf_key in list(constraintDict.keys()):
            for constraint_data in upperDict[leaf_key]:
                bound = float(np.exp(constraint_data[0]))
                mono = np.prod([ivars[j] ** constraint_data[1][j]
                                for j in range(len(constraint_data[1]))]).value
                constraintDict[leaf_key].append(bound ** (1.-epsilon) <= mono)
            for constraint_data in lowerDict[leaf_key]:
                bound = float(np.exp(constraint_data[0]))
                mono = np.prod([ivars[j] ** constraint_data[1][j]
                                for j in range(len(constraint_data[1]))]).value
                constraintDict[leaf_key].append(bound ** (1.+epsilon) >= mono)
        return constraintDict

    @staticmethod
    def pwl_constraintify(pwlDict, dvar, ivars, oper='>='):
        """
        Turns pwl constraint data into monomial inequality constraints
        Arguments:
            pwlDict: Dict[leaf_number] containing [B0 (offset), B (linear)]
            dvar: dependent monomial
            ivars: independent monomials
            oper: type of inequality (default lower bounds dvar)
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
            try:
                constraintDict[key] = [PosynomialInequality(dvar, oper, c * mono)]
            except:
                print("Culprits are: \n %s, \n %s, and \n %s ..." %
                      (dvar, oper, c*mono))
        return constraintDict

    @staticmethod
    def bigM_constraintify(upperDict, lowerDict, pwlDict,
                           dvar, ivars, oper='>=', M = 10):
        if oper == '=':
            raise ValueError('Equalities are not supported for big-M '
                             'formulations. Please reformulate and try again.')
        binary_var = VectorVariable(len(pwlDict), 'bin', '-', 'binary variable', bin=True)
        leaves = list(pwlDict.keys())
        pwlConstraints = ConstraintTree.pwl_constraintify(pwlDict, dvar,
                                                          ivars, oper=oper)
        trConstraints = ConstraintTree.tr_constraintify(upperDict, lowerDict, ivars)
        constraintDict = mergeDict(pwlConstraints, trConstraints)
        for i in range(len(leaves)):
            for constraint in constraintDict[leaves[i]]:
                if constraint.oper == '>=':
                    bigMconstr = PosynomialInequality(constraint.left * binary_var[i]**M,
                                                      constraint.oper, constraint.right)
                elif constraint.oper == '<=':
                    bigMconstr = PosynomialInequality(constraint.left, constraint.oper,
                                                      constraint.right * binary_var[i]**M)
                constraintDict[leaves[i]].append(bigMconstr)
        return constraintDict

    @staticmethod
    def binary_constraintify(bin_vectorvar, solve_type):
        """ Generates list of constraints that ensure that binary variables
        are valued either 1 or e. """
        constraints = []
        if solve_type == 'mi':
            constraints.append(np.prod(bin_vectorvar) == np.exp(1))
        if solve_type == 'global':
            constraints.append(np.prod(bin_vectorvar) == np.exp(1))
            for i in range(len(bin_vectorvar)):
                var = bin_vectorvar[i]
                with SignomialsEnabled():
                    constraints.append(SignomialEquality(var**2 - (1+np.exp(1))*var + np.exp(1), 0))
        return constraints



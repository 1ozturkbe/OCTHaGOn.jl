import pandas as pd
import numpy as np
from interpretableai import iai
# Check out https://docs.interpretable.ai/stable/IAI-Python/installation/
# for in depth installation info
from gpkit import Variable, Monomial
from gpkit.varkey import VarKey


# Using trees to obtain PWL approximations with trust regions
# Each leaf has a set of PWL constraints (hyperplanes of the form β0 + β'x <= y)
# as well as trust regions (threshold <= α for upper split, threshold >= α for lower split)

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
        vks = ['x' + str(i) for i in range(1, n_nodes + 1)]
    all_leaves = [i for i in range(1, n_nodes + 1) if lnr.is_leaf(i)]  # Julia is one-indexed!
    pwlDict = {leaf: [] for leaf in all_leaves}
    for i in range(len(all_leaves)):
        β0 = lnr.get_regression_constant(all_leaves[i]);
        weights = lnr.get_regression_weights(all_leaves[i])[0];
        β = [];
        for j in range(len(vks)):
            if vks[j] in weights.keys():
                β.append(weights[vks[j]]);
            else:
                β.append(0.);
        pwlDict[all_leaves[i]] = [β0, β]
    return pwlDict


def monomials_from_pwl_data(pwlDict, gpvars):
    """
    Creates monomials from pwlDict over GPkit variables
    Arguments:
        pwlDict: Dict[leaf_number] containing [B0 (offset), B (linear)]
        gpvars: list of GPkit.variables
    Returns:
        Dict[leaf_number] containing monomial
    """
    constraintDict = {}
    for key, value in pwlDict.items():
        c = np.exp(value[0])
        if len(gpvars) != len(value[1]):
            raise ValueError("Number of GP variables don't match "
                             "approximation dimension.")
        exponential = np.prod([gpvars[i] ** value[1][i]
                               for i in range(len(gpvars))])
        constraintDict[key] = c * exponential
    return constraintDict


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
        vks = ['x' + str(i) for i in range(1, n_nodes + 1)]
    all_leaves = [i for i in range(1, n_nodes + 1) if lnr.is_leaf(i)]  # Julia is one-indexed!
    upperDict = {leaf: [] for leaf in all_leaves}
    lowerDict = {leaf: [] for leaf in all_leaves}
    for i in range(len(all_leaves)):
        # Find all parents
        parents = [all_leaves[i]];
        while lnr.get_depth(parents[-1]) > 0:
            parents.append(lnr.get_parent(parents[-1]))
        for j in parents[1:]:
            # For each parent, define trust region with binary variables
            threshold = lnr.get_split_threshold(j)
            if lnr.is_hyperplane_split(j):
                weights = lnr.get_split_weights(j)[0];
            else:
                feature = lnr.get_split_feature(j);
                weights = {feature: 1};
            upper = lnr.get_upper_child(j) in parents  # Checking upper vs. lower split
            α = []
            for k in range(len(vks)):
                if vks[k] in weights.keys():
                    α.append(weights[vks[k]]);
                else:
                    α.append(0.);
            if upper:
                upperDict[all_leaves[i]].append([threshold, α])
            else:
                lowerDict[all_leaves[i]].append([threshold, α])
    return upperDict, lowerDict

def bounding_constraints(bounds, m):
    """ Provides bounding constraints to a GP based on the bounds on
    the fitted variables. """
    pass

# def sequential_trust_region(upperDict, lowerDict, gpvars, epsilon=1e-3):
#     trDict = {key: [] for key, _ in upperDict.items()}
#     for key in upperDict.items():
#
#
#     return trDict


# def signomial_trust_region(upperDict, lowerDict, gpvars):
# #     for i in upperDict:
#     return

if __name__ == "__main__":
    pass

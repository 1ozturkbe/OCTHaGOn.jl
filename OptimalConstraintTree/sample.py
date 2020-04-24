"""
These methods use the pyDOE package to generate samples from unknown or known functions.
"""
import numpy as np
from pyDOE import lhs
from gpkit import SignomialsEnabled, units
from gpkit.small_scripts import mag
from gpkit.exceptions import InvalidGPConstraint, Infeasible
from gpkit.exceptions import UnknownInfeasible
from gpkit import Monomial
from gpkit.keydict import KeyDict
import progressbar
from time import sleep

from OptimalConstraintTree.tools import clean_subs
from OptimalConstraintTree.tools import blockPrint, enablePrint

def sample_gpobj(gpobj, bounds, samples, criterion="corr"):
    """
    Latin hypercube samples a given GPkit nomial object.
    :param gpobj: GPkit nomial object
    :param samples: number of samples
    :param criterion: (from pyDOE documentation)
            “center” or “c”: center the points within the sampling intervals
            “maximin” or “m”: maximize the minimum distance between points,
                but place the point in a randomized location within its interval
            “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
            “correlation” or “corr”: minimize the maximum correlation coefficient
    :return: Y and X data for the gpobj
    """
    vks = list(gpobj.varkeys)
    #blockPrint()
    levels = lhs(len(vks), samples=samples, criterion=criterion)
    #enablePrint()
    levels = [dict(zip(vks, levels[i])) for i in range(samples)]
    Xsubs = []
    Yvals = []
    for i in range(samples):
        next_subs = {vk: bounds[vk][0] +
                          (bounds[vk][1] - bounds[vk][0]) * levels[i][vk]
                      for vk in vks}
        Xsubs.append(clean_subs(next_subs))
        Yvals.append(gpobj.sub(Xsubs[-1]))
    return Yvals, Xsubs

def sample_gpmodel(gpmodel, bounds, samples, criterion='corr', verbosity=0):
    """
    Latin hypercube sampling of a GP model within given input bounds.
    :param gpmodel: A GPkit model
    :param bounds: dictionary of bounds for varkeys of interest
    :param samples: number of samples
    :param criterion: (from pyDOE documentation)
            “center” or “c”: center the points within the sampling intervals
            “maximin” or “m”: maximize the minimum distance between points,
                but place the point in a randomized location within its interval
            “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
            “correlation” or “corr”: minimize the maximum correlation coefficient
    :param verbosity:
    :return: solutions and corresponding substitutions
    """
    vks = list(bounds.keys())
    # blockPrint()
    levels = lhs(len(vks), samples=samples, criterion=criterion)
    # enablePrint()
    levels = [dict(zip(vks, levels[i])) for i in range(samples)]
    subs = []
    solns = []
    print("\nSampling model %s over %s samples..." % (type(gpmodel), str(samples)))
    bar = progressbar.ProgressBar(maxval=samples, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(samples):
        with SignomialsEnabled():
            next_subs = clean_subs({vk: bounds[vk][0] +
                          (bounds[vk][1] - bounds[vk][0]) * levels[i][vk]
                      for vk in vks})
        gpmodel.substitutions.update(next_subs)
        try:
            solns.append(gpmodel.solve(verbosity=verbosity))
            subs.append(next_subs)
        except InvalidGPConstraint:
            solns.append(gpmodel.localsolve(verbosity=verbosity))
            subs.append(next_subs)
        except (Infeasible, UnknownInfeasible):
            print("Model infeasible at substitutions %s" % next_subs)
        bar.update(i+1)
    bar.finish()
    return solns, subs

def gen_X(subs, basis):
    """
    Generates X data with normalization using the basis.
    Also strips monomials.
    :param subs: List of subs dicts
    :param basis: subs dict with
    :return: X, a numpy array of data
    """
    n_samples = len(subs)
    keys = list(basis.keys())
    subs = [KeyDict(sub) for sub in subs]
    basis = KeyDict(basis)
    X = np.zeros((n_samples, len(keys)))
    try:
        for i in range(n_samples):
            for j in range(len(keys)):
                sub = subs[i][keys[j].name]
                base = basis[keys[j].name]
                if type(base) is Monomial:
                    base = base.value
                if type(sub) is Monomial:
                    sub = sub.value
                try:
                    X[i,j] = (sub/base).to('')
                except AttributeError:
                    X[i,j] = sub/base
                except ValueError:
                    print("Units of data point [%s, %s] "
                                     "don't match." % (i, keys[j]))
    except KeyError:
        raise KeyError("Substitutions and the basis have different "
                         "keys. Please make uniform (Variable preferred) "
                         "and try again.")
    return X

if __name__ == "__main__":
    pass

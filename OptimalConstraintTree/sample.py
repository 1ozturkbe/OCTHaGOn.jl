"""
These methods use the pyDOE package to generate samples from unknown or known functions.
"""
from pyDOE import lhs
from gpkit import SignomialsEnabled, units
from gpkit.exceptions import InvalidGPConstraint, Infeasible
from gpkit.exceptions import UnknownInfeasible

from OptimalConstraintTree.tools import clean_subs

def sample_gpobj(gpobj, bounds, samples, criterion="corr"):
    """
    Latin hypercube samples a given GPkit nomial object.
    :param gpobj: GPkit nomial object
    :param samples: Number of samples desired
    :param criterion: (from pyDOE documentation)
            “center” or “c”: center the points within the sampling intervals
            “maximin” or “m”: maximize the minimum distance between points,
                but place the point in a randomized location within its interval
            “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
            “correlation” or “corr”: minimize the maximum correlation coefficient
    :return: Y and X data for the gpobj
    """
    vks = list(gpobj.varkeys)
    levels = lhs(len(vks), samples=samples, criterion=criterion)
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
    vks = list(bounds.keys())
    levels = lhs(len(vks), samples=samples, criterion=criterion)
    levels = [dict(zip(vks, levels[i])) for i in range(samples)]
    subs = []
    solns = []
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
    return solns, subs

if __name__ == "__main__":
    pass

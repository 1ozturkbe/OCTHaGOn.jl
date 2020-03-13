"""
These methods use the pyDOE package to generate samples from unknown or known functions.
"""
from pyDOE import lhs

def lh_sample_gpobj(gpobj, bounds, samples, criterion="corr"):
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
    Xsubs = []
    Yvals = []
    for i in range(samples):
        Xsubs.append({vks[j]: bounds[vks[j]][0] +
                              (bounds[vks[j]][1] - bounds[vks[j]][0]) * levels[i][j]
                      for j in range(len(vks))})
        Yvals.append(gpobj.sub(Xsubs[-1]))
    return Yvals, Xsubs

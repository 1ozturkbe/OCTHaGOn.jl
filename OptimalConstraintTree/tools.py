import progressbar
from time import sleep
import sys, os

from gpfit.fit_constraintset import FitCS

# Disable or enable printing for problematic functions
def blockPrint():
    """ Disables printing from this onward.
    Printing MUST BE REACTIVATED after. """
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """ Enables print outputs.
    Counteracts blockPrint. """
    sys.stdout = sys.__stdout__

def clean_subs(subs):
    """
    Takes messy subs dicts and cleans it up.
    :param subs: Substitutions dictionary
    :return: Dict with proper uniting
    """
    cleaned_subs = {}
    vks = subs.keys()
    for vk in vks:
        try:
            cleaned_subs[vk] = subs[vk].value
        except:
            cleaned_subs[vk] = subs[vk]
    return cleaned_subs

def find_signomials(model):
    """
    Finds are returns all signomials in a GPkit model
    :param model:
    :return:
    """
    pass

def constraint_from_gpfit(fitcs, ivar, dvars, basis=None):
    """
    Returns a posynomial constraint from a data fit
    :param fitcs: FitCS object
    :param ivar: independent variable
    :param dvars: dependent variables
    :return: constraint
    """
    basis_keys = list(basis.keys())
    dvar_keys = [dvar.key for dvar in dvars]
    try:
        nondim_ivar = ivar/basis[ivar.key]
    except KeyError:
        raise KeyError("Independent variable %s "
                       "is not in the basis to be "
                       "normalized" % ivar.key)
    basis.pop(ivar.key)
    try:
        nondim_dvars = [dvars[i]/basis[basis_keys[i]] for i in range(len(basis.keys()))]
    except KeyError:
        raise KeyError("There is a dependent variable "
                       "missing in the basis, or there is an "
                       "incorrect type. ")
    new_fitcs = FitCS(fitcs.fitdata, nondim_ivar, nondim_dvars)
    return new_fitcs

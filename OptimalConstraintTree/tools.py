import progressbar
from time import sleep
import sys, os
import io

from gpfit.fit_constraintset import FitCS

text_trap = io.StringIO()

# Disable or enable printing for problematic functions
def blockPrint():
    """ Disables printing from this onward.
    Printing MUST BE REACTIVATED after. """
    sys.stdout = text_trap

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

def constraint_from_gpfit(fitcs, dvar, ivars, basis=None):
    """
    Returns a posynomial constraint from a data fit
    :param fitcs: FitCS object
    :param dvar: dependent variable
    :param ivars: independent variables
    :return: constraint
    """
    basis_keys = list(basis.keys())
    ivar_keys = [ivar.key for ivar in ivars]
    try:
        nondim_dvar = dvar/basis[dvar.key]
    except KeyError:
        raise KeyError("Independent variable %s "
                       "is not in the basis to be "
                       "normalized" % dvar.key)
    basis.pop(dvar.key)
    try:
        nondim_ivars = [ivars[i]/basis[basis_keys[i]] for i in range(len(basis.keys()))]
    except KeyError:
        raise KeyError("There is a dependent variable "
                       "missing in the basis, or there is an "
                       "incorrect type. ")
    new_fitcs = FitCS(fitcs.fitdata, nondim_dvar, nondim_ivars)
    return new_fitcs

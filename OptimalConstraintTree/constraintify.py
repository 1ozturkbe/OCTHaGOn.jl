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


def bounding_constraints(bounds, m):
    """ Provides bounding constraints to a GP based on the bounds on
    the fitted variables. """
    pass

if __name__ == "__main__":
    pass

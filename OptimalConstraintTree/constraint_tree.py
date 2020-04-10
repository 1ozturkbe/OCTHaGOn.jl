from gpkit import Variable, Model
from gpkit.small_scripts import mag

import numpy as np

from constraintify import pwl_constraint_data, trust_region_data


class ConstraintTree:
    """ ConstraintTree class, containing the approximate constraints
    from ORT models"""

    def __init__(self, lnr, vars, vks=None, type='seq'):
        self.learner = lnr  # original PWL learner
        self.learner_units = None
        self.type = type
        self.vars = vars
        self.vks = vks
        self.learner_params = self.learner.params()
        self.nodes = np.range(1, self.learner.get_num_nodes() + 1)
        self.leaves = [i for i in self.nodes if self.learner.is_leaf(i)]
        self.constraints = {}

    def setup(self):
        # Generating constraint data from ORT
        self.pwl_data = pwl_constraint_data(self.learner, self.vks)
        self.tr_data = trust_region_data(self.learner, self.vks)
        if self.type == "seq":
            pass
        elif self.type == "mi":
            pass
        elif self.type == "global":
            pass
        else:
            raise ValueError("ConstraintTree type %s "
                             "is not supported." % self.type)

    def set_units(self, units):
        """ Makes sure that constraint tree is evaluated with the
            in the same units as the training data. """
        pass

    def get_leaf_constraint(self, sol):
        """ Returns constraints for a particular leaf
        depending on the features (solution of free vars). """
        inps = np.array([np.log(mag(sol))])
        leaf_no = self.learner.apply(inps)
        return self.constraints[leaf_no]

    def piecewise_convexify(self, rms_threshold=0.01, max_threshold=0.02):
        """ Prunes the tree using by efficiently computing adjacent convex regions,
        returning a new tree. """
        pass

# class GlobalModel(Model):
#     def __init__(self, model, constraint_trees):
#
#     def setup(self):
#         return None

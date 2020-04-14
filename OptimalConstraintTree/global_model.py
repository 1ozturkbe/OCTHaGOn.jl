# class GlobalModel(Model):
#     """ Extends the GPkit Model class to be able to accommodate
#     constraint trees.
#
#     Arguments
#     ---------
#     cost : Posynomial
#
#     constraints : ConstraintSet or list of constraints or ConstraintTrees.
#
#     substitutions : dict
#         Note: Do not forget substitutions required for ConstraintTrees and
#         GP object fits.
#     """
#     def __init__(self, cost, constraints, *args, **kwargs):
#         self.trees = kwargs.pop("trees")
#         self.treevars = set([[tree.ivar, dvar for dvar in tree.ivars]
#                             for tree in self trees])
#         self.spmodel = Model.__init__(cost, constraints, *args, **kwargs)
#
#     def globalsolve(self, verbosity=0):
#         pass

#=
optimize:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-26
Optimizes GlobalModels using all methods.
=#

"""
    globalsolve(gm::GlobalModel)

Creates and solves the global optimization model using the linear constraints from GlobalModel,
and approximated nonlinear constraints from inside its BlackBoxFunctions.
"""
function globalsolve(gm::GlobalModel)
    clear_tree_constraints!(gm)   # remove trees from previous solve (if any).
    add_tree_constraints!(gm)     # refresh latest tree constraints.
    status = optimize!(gm.model)
    return status
end
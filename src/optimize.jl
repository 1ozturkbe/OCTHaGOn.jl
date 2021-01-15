"""
    globalsolve(gm::GlobalModel)

Creates and solves the global optimization model using the linear constraints from GlobalModel,
and approximated nonlinear constraints from inside its BlackBoxLearners.
"""
function globalsolve(gm::GlobalModel)
    clear_tree_constraints!(gm)   # remove trees from previous solve (if any).
    add_tree_constraints!(gm)     # refresh latest tree constraints.
    status = JuMP.optimize!(gm)
    return status
end
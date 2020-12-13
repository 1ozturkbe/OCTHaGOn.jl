""" Clears the constraints in GM of bbf.mi_constraints. """
function clear_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxFunction})
    for bbf in bbfs
        for constraint in bbf.mi_constraints
            if is_valid(gm.model, constraint)
                delete(gm.model, constraint)
            end
        end
        for variable in bbf.leaf_variables
            if is_valid(gm.model, variable)
                delete(gm.model, variable)
            end
        end
    end
    return
end

function clear_tree_constraints!(gm::GlobalModel)
    clear_tree_constraints!(gm, gm.bbfs)
    return
end

"""
    add_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxFunction}; M=1e5)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxFunction}; M=1e5)
    for bbf in bbfs
        # Battery of checks
        if bbf.settings[:reloaded]
            bbf.mi_constraints, bbf.leaf_variables = add_feas_constraints!(gm.model, bbf.vars,
                                            bbf.learners[end].lnr;
                                            M=M, eq=bbf.equality, return_data = true);
        elseif bbf.feas_ratio == 1.0
            continue
        elseif size(bbf.X, 1) == 0
            throw(OCTException("Constraint " * string(bbf.name) * " has not been sampled yet, and is thus untrained."))
        elseif length(bbf.learners) == 0
            throw(OCTException("Constraint " * string(bbf.name) * " has not been learned yet"))
        elseif bbf.feas_ratio == 0.0
            throw(OCTException("Constraint " * string(bbf.name) * " is INFEASIBLE but you tried to include it in
                   your global problem. Find at least one feasible solution, train and try again."))
        elseif isempty(bbf.learners)
            throw(OCTException("Constraint " * string(bbf.name) * " must be learned before tree constraints
                                can be generated."))
        elseif !gm.settings[:ignore_accuracy] && !check_accuracy(bbf)
            throw(OCTException("Constraint " * string(bbf.name) * " is inaccurately approximated. "))
        else
            bbf.mi_constraints, bbf.leaf_variables = add_feas_constraints!(gm.model, bbf.vars, bbf.learners[end].lnr;
                                                       M=M, eq=bbf.equality, return_data = true);
        end
    end
    return
end

function add_tree_constraints!(gm::GlobalModel; M=1e5)
    add_tree_constraints!(gm, gm.bbfs, M=M)
    return
end

"""
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        lnr: A fitted IAI.OptimalTreeLearner
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
"""
function add_feas_constraints!(m::JuMP.Model, x, lnr::IAI.OptimalTreeLearner;
                               M::Float64 = 1.e5, eq = false,
                               return_data::Bool = false)
    check_if_trained(lnr);
    n_nodes = IAI.get_num_nodes(lnr);
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)];
    feas_leaves =
        [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    infeas_leaves = [i for i in all_leaves if i ∉ feas_leaves];
    constraints = []; leaf_variables = [];
    z_feas = @variable(m, [1:size(feas_leaves, 1)], Bin)
    append!(leaf_variables, z_feas)
    push!(constraints, @constraint(m, sum(z_feas) == 1))
    if eq
        z_infeas = @variable(m, [1:length(infeas_leaves)], Bin)
        append!(leaf_variables, z_infeas)
        push!(constraints, @constraint(m, sum(z_infeas) == 1))
    end
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
    for i = 1:size(feas_leaves, 1)
        leaf = feas_leaves[i]
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            push!(constraints, @constraint(m, threshold <= sum(α .* x) + M * (1 - z_feas[i])))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            push!(constraints, @constraint(m, threshold + M * (1 - z_feas[i]) >= sum(α .* x)))
        end
    end
    if eq
        for i = 1:size(infeas_leaves, 1)
            leaf = infeas_leaves[i]
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                push!(constraints, @constraint(m, threshold <= sum(α .* x) + M * (1 - z_infeas[i])))
            end
            for region in lowerDict[leaf]
                threshold, α = region
                push!(constraints, @constraint(m, threshold + M * (1 - z_infeas[i]) >= sum(α .* x)))
            end
        end
    end
    if return_data
        return constraints, leaf_variables
    else
        return
    end
end

function add_regr_constraints!(m::JuMP.Model, x::Array, y, grid::IAI.GridSearch;
                               M::Float64 = 1.e5, eq = false,
                               return_data::Bool = false)
    """
    Creates a set of MIO constraints from a OptimalTreeRegressor
    Arguments:
        m:: JuMP Model
        x:: independent JuMPVariable (features in lnr)
        y:: dependent JuMPVariable (output of lnr)
        grid: A fitted Grid
        M:: coefficient in bigM formulation
    """
    lnr = IAI.get_learner(grid)
    check_if_trained(lnr)
    n_nodes = IAI.get_num_nodes(lnr)
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    constraints = []; leaf_variables = [];
    z = @variable(m, [1:size(all_leaves, 1)], Bin)
    append!(leaf_variables, z)
    push!(constraints, @constraint(m, sum(z) == 1))
    # Getting lnr data
    pwlDict = pwl_constraint_data(lnr, Symbol.(x))
    upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
    for i = 1:size(all_leaves, 1)
        # ADDING CONSTRAINTS
        leaf = all_leaves[i]
        β0, β = pwlDict[leaf]
        push!(constraints, @constraint(m, sum(β .* x) + β0 <= y + M * (1 .- z[i])))
        if eq
            push!(constraints, @constraint(m, sum(β .* x) + β0 + M * (1 .- z[i]) >= y))
        end
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            push!(constraints, @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i])))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            push!(constraints, @constraint(m, threshold + M * (1 - z[i]) >= sum(α .* x)))
        end
    end
    if return_data
        return constraints, leaf_variables
    else
        return
    end
end

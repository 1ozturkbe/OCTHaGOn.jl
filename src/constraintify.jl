""" 
    clear_tree_constraints!(gm::GlobalModel, bbf::{BlackBoxClassifier, BlackBoxRegressor})
    clear_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxClassifier, BlackBoxRegressor})
    clear_tree_constraints!(gm::GlobalModel)

Clears the constraints bbf.mi_constraints 
    as well as bbf.leaf_variables in GlobalModel. 
"""
function clear_tree_constraints!(gm::GlobalModel, bbf::Union{BlackBoxClassifier, BlackBoxRegressor})
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
    return
end

function clear_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxClassifier, BlackBoxRegressor})
    for bbf in bbfs
        clear_tree_constraints!(gm, bbf)
    end
    return
end

clear_tree_constraints!(gm::GlobalModel) = clear_tree_constraints!(gm, gm.bbfs)

"""
    add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier; M=1e5)
    add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor; M=1e5)
    add_tree_constraints!(gm::GlobalModel, bbfs::Array; M=1e5)
    add_tree_constraints!(gm::GlobalModel)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier; M = 1e5)
    if bbc.feas_ratio == 1.0
        return
    elseif size(bbc.X, 1) == 0 && !get_param(bbc, :reloaded)
        throw(OCTException("Constraint " * string(bbc.name) * " has not been sampled yet, and is thus untrained."))
    elseif length(bbc.learners) == 0
        throw(OCTException("Constraint " * string(bbc.name) * " has not been learned yet"))
    elseif bbc.feas_ratio == 0.0 && !get_param(bbc, :reloaded)
        throw(OCTException("Constraint " * string(bbc.name) * " is INFEASIBLE but you tried to include it in
            your global problem. Find at least one feasible solution, train and try again."))
    elseif isempty(bbc.learners)
        throw(OCTException("Constraint " * string(bbc.name) * " must be learned before tree constraints
                            can be generated."))
    elseif !get_param(gm, :ignore_accuracy) && !check_accuracy(bbc)
        throw(OCTException("Constraint " * string(bbc.name) * " is inaccurately approximated. "))
    else
        bbc.mi_constraints, bbc.leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[end].lnr;
                                            M=M, equality = bbc.equality);
    end
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor; M = 1e5)
    # TODO: check regressions much better
    bbr.mi_constraints, bbr.leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                                   bbr.learners[end].lnr;
                                                                   M = M, equality = bbr.equality)
    return
end

function add_tree_constraints!(gm::GlobalModel, bbfs::Array{BlackBoxClassifier, BlackBoxRegressor}; M = 1e5)
    for bbf in bbfs
        add_tree_constraints!(gm, bbf; M = M)
    end
    return
end

add_tree_constraints!(gm::GlobalModel; M = 1e5) = add_tree_constraints!(gm, gm.bbfs; M = M)

"""
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        lnr: A fitted IAI.OptimalTreeLearner
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
"""
function add_feas_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, lnr::IAI.OptimalTreeClassifier;
                               M::Float64 = 1.e5, equality::Bool = false)
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
    if equality
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
    if equality
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
    return constraints, leaf_variables
end

function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeRegressor;
                               M::Float64 = 1.e5, equality::Bool = false)
    """
    Creates a set of MIO constraints from a OptimalTreeRegressor
    Arguments:
        m:: JuMP Model
        x:: independent JuMPVariable (features in lnr)
        y:: dependent JuMPVariable (output of lnr)
        grid: A fitted Grid
        M:: coefficient in bigM formulation
    """
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
        if equality
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
    return constraints, leaf_variables
end

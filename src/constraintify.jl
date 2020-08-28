# File to generate constraints compatible with JuMP
function check_if_trained(lnr::IAI.OptimalTreeLearner)
    try
        n_nodes = IAI.get_num_nodes(lnr);
    catch err
        if isa(err, UndefRefError)
            throw(OCTException("Grids/trees require training before being used in constraints!"))
        else
            rethrow(err)
        end
    end
end

function add_tree_constraints!(md::ModelData; M=1e5)
    for bbf in md.fns
        if bbf.feas_ratio == 1.0
            return
        elseif bbf.feas_ratio == 0.0
            @warn("Constraint " * string(bbf.name) * " is INFEASIBLE but you tried to include it in
                   your global problem. For now, the constraint is OMITTED.
                   Find at least one feasible solution, train and try again.")
        elseif isempty(bbf.learners)
            throw(OCTException("Constraint " * string(bbf.name) * " must be learned before tree constraints
                                can be generated."))
        else
            grid = bbf.learners[end];
            constrs = add_feas_constraints!(md.JuMP_model,
                                        [md.JuMP_vars[vk] for vk in bbf.vks],
                                        bbf.learners[end], bbf.vks;
                              M=M, eq=bbf.equality,
                              return_constraints = true);
            bbf.constraints = constrs; #TODO: make sure we can remove all constraints.
        end
    end
    return
end

function add_feas_constraints!(m::JuMP.Model, x, grid::IAI.GridSearch,
                               vks::Array; M::Float64 = 1.e5, eq = false,
                               return_constraints::Bool = false)
    """
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        grid: A fitted GridSearch
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
        vks:: varkeys of the features in lnr
    """
    #TODO determine proper use for equalities
    lnr = IAI.get_learner(grid);
    check_if_trained(lnr);
    n_nodes = IAI.get_num_nodes(lnr);
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)];
    feas_leaves =
        [i for i in all_leaves if IAI.get_classification_label(lnr, i)];
    infeas_leaves = [i for i in all_leaves if i ∉ feas_leaves];
    z_feas = @variable(m, [1:size(feas_leaves, 1)], Bin)
    count = 0; constraints = Dict()
    constraints[count += 1] = @constraint(m, sum(z_feas) == 1)
    if eq
        z_infeas = @variable(m, [1:length(infeas_leaves)], Bin)
        constraints[count += 1] = @constraint(m, sum(z_infeas) == 1)
    end
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, vks)
    for i = 1:size(feas_leaves, 1)
        leaf = feas_leaves[i]
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            constraints[count += 1] = @constraint(m, threshold <= sum(α .* x) + M * (1 - z_feas[i]))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            constraints[count += 1] = @constraint(m, threshold + M * (1 - z_feas[i]) >= sum(α .* x))
        end
    end
    if eq
        for i = 1:size(infeas_leaves, 1)
            leaf = infeas_leaves[i]
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                constraints[count += 1] = @constraint(m, threshold <= sum(α .* x) + M * (1 - z_infeas[i]))
            end
            for region in lowerDict[leaf]
                threshold, α = region
                constraints[count += 1] = @constraint(m, threshold + M * (1 - z_infeas[i]) >= sum(α .* x))
            end
        end
    end
    if return_constraints
        return constraints
    else
        return
    end
end

function add_regr_constraints!(m::JuMP.Model, x::Array, y, grid::IAI.GridSearch, vks::Array;
                               M::Float64 = 1.e5, eq = false,
                               return_constraints::Bool = false)
    """
    Creates a set of MIO constraints from a OptimalTreeRegressor
    Arguments:
        m:: JuMP Model
        x:: independent JuMPVariable (features in lnr)
        y:: dependent JuMPVariable (output of lnr)
        grid: A fitted Grid
        vks:: varkeys of the features in lnr
        M:: coefficient in bigM formulation
    """
    #TODO determine proper use for equalities
    lnr = IAI.get_learner(grid);
    check_if_trained(lnr);
    n_nodes = IAI.get_num_nodes(lnr)
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    z = @variable(m, [1:size(all_leaves, 1)], Bin)
    count = 0; constraints = Dict()
    constraints[count += 1] = @constraint(m, sum(z) == 1)
    # Getting lnr data
    pwlDict = pwl_constraint_data(lnr, vks)
    upperDict, lowerDict = trust_region_data(lnr, vks)
    for i = 1:size(all_leaves, 1)
        # ADDING CONSTRAINTS
        leaf = all_leaves[i]
        β0, β = pwlDict[leaf]
        constraints[count += 1] = @constraint(m, sum(β .* x) + β0 <= y + M * (1 .- z[i]))
        if eq
            constraints[count += 1] = @constraint(m, sum(β .* x) + β0 + M * (1 .- z[i]) >= y)
        end
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            constraints[count += 1] = @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i]))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            constraints[count += 1] = @constraint(m, threshold + M * (1 - z[i]) >= sum(α .* x))
        end
    end
    if return_constraints
        return constraints
    else
        return
    end
end

function pwl_constraint_data(lnr, vks)
    """
    Creates PWL dataset from a OptimalTreeLearner
    Arguments:
        lnr: OptimalTreeLearner
        vks: headers of DataFrame X, i.e. varkeys
    Returns:
        Dict[leaf_number] containing [B0, B]
    """
    n_nodes = IAI.get_num_nodes(lnr)
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    pwlConstraintDict = Dict()
    for i = 1:size(all_leaves, 1)
        β0 = IAI.get_regression_constant(lnr, all_leaves[i])
        weights = IAI.get_regression_weights(lnr, all_leaves[i])[1]
        β = []
        for i = 1:size(vks, 1)
            if vks[i] in keys(weights)
                append!(β, weights[vks[i]])
            else
                append!(β, 0.0)
            end
        end
        pwlConstraintDict[all_leaves[i]] = [β0, β]
    end
    return pwlConstraintDict
end

function trust_region_data(lnr, vks)
    """
    Creates trust region from a OptimalTreeLearner
    Arguments:
        lnr: OptimalTreeLearner
        vks: headers of DataFrame X, i.e. varkeys
    Returns:
        Dict[leaf_number] containing [B0, B]
    """
    n_nodes = IAI.get_num_nodes(lnr)
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    upperDict = Dict()
    lowerDict = Dict()
    for i = 1:size(all_leaves, 1)
        # Find all parents
        parents = [all_leaves[i]]
        while IAI.get_depth(lnr, parents[end]) > 0
            append!(parents, IAI.get_parent(lnr, parents[end]))
        end
        upperDict[all_leaves[i]] = []
        lowerDict[all_leaves[i]] = []
        for j in parents[2:end]
            # For each parent, define trust region with binary variables
            threshold = IAI.get_split_threshold(lnr, j)
            if IAI.is_hyperplane_split(lnr, j)
                weights = IAI.get_split_weights(lnr, j)
                weights = weights[1]
            else
                feature = IAI.get_split_feature(lnr, j)
                weights = Dict(feature => 1)
            end
            upper = IAI.get_upper_child(lnr, j) in parents # Checking upper vs. lower split
            α = []
            for i = 1:size(vks, 1)
                if vks[i] in keys(weights)
                    append!(α, weights[vks[i]])
                else
                    append!(α, 0.0)
                end
            end
            if upper
                append!(upperDict[all_leaves[i]], [[threshold, α]])
            else
                append!(lowerDict[all_leaves[i]], [[threshold, α]])
            end
        end
    end
    return upperDict, lowerDict
end

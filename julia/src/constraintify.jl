# File to generate constraints compatible with JuMP
using JuMP
using Gurobi

function base_otr()
    return IAI.OptimalTreeRegressor(
        random_seed = 1,
        max_depth = 3,
        cp = 1e-10,
        minbucket = 0.03,
        regression_sparsity = :all,
        fast_num_support_restarts = 1,
        hyperplane_config = (sparsity = :1,),
        regression_lambda = 0.00001,
        regression_weighted_betas = true,
    )
end

function base_otc()
    return IAI.OptimalTreeClassifier(
        random_seed = 1,
        max_depth = 5,
        cp = 1e-10,
        minbucket = 0.01,
        fast_num_support_restarts = 1,
        hyperplane_config = (sparsity = :1,),
    )
end

function learn_constraints(lnr, constraints, X; idxs = nothing)
    """
    Returns a set of feasibility trees from a set of constraints.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        constraints: set of constraint functions in std form (>= 0)
        X: samples of the free variables
    NOTE: All constraints must take in vector of all X values.
    """
    n_samples, n_features = size(X)
    n_constraints = size(constraints, 1)
    feasTrees = IAI.OptimalTreeClassifier[]
    for i = 1:n_constraints
        Y = [constraints[i](X[j, :]) >= 0 for j = 1:n_samples]
        # Making sure that we only consider relevant features.
        if !isnothing(idxs)
            IAI.set_params!(lnr, split_features = idxs[i])
            if typeof(lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(lnr, regression_features=idxs[i])
            end
        else
            IAI.set_params!(lnr, split_features = :all)
            if typeof(lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(lnr, regression_features=:all)
            end
        end
        IAI.fit!(lnr, X, Y)
        append!(feasTrees, [lnr])
    end
    return feasTrees
end

function bound_variables(m, x, lbs, ubs)
    for i = 1:size(lbs, 1)
        @constraint(m, x[i] <= ubs[i])
        @constraint(m, x[i] >= lbs[i])
    end
    return m
end

function add_feas_constraints(lnr, m, x, vks; M = 1e5)
    """
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        lnr: OptimalTreeClassifier
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
        vks:: varkeys of the features in lnr
    """
    n_nodes = IAI.get_num_nodes(lnr)
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    feas_leaves =
        [i for i in all_leaves if IAI.get_classification_label(lnr, i)]
    z = @variable(m, [1:size(feas_leaves, 1)], Bin)
    @constraint(m, sum(z) == 1)
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, vks)
    for i = 1:size(feas_leaves, 1)
        leaf = feas_leaves[i]
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i]))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i]))
        end
    end
    return m
end

function add_mio_constraints(lnr, m, x, y, vks; M = 1e5, eq = false)
    """
    Creates a set of MIO constraints from a OptimalTreeRegressor
    Arguments:
        lnr: OptimalTreeRegressor
        m:: JuMP Model
        x:: independent JuMPVariable (features in lnr)
        y:: dependent JuMPVariable (output of lnr)
        vks:: varkeys of the features in lnr
        M:: coefficient in bigM formulation
    """
    n_nodes = IAI.get_num_nodes(lnr)
    # Add a binary variable for each leaf
    all_leaves = [i for i = 1:n_nodes if IAI.is_leaf(lnr, i)]
    z = @variable(m, [1:size(all_leaves, 1)], Bin)
    @constraint(m, sum(z) == 1)
    # Getting lnr data
    pwlDict = pwl_constraint_data(lnr, vks)
    upperDict, lowerDict = trust_region_data(lnr, vks)
    for i = 1:size(all_leaves, 1)
        # ADDING CONSTRAINTS
        leaf = all_leaves[i]
        β0, β = pwlDict[leaf]
        @constraint(m, sum(β .* x) + β0 <= y + M * (1 .- z[i]))
        if eq
            @constraint(m, sum(β .* x) + β0 + M * (1 .- z[i]) >= y)
        end
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i]))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i]))
        end
    end
    return m
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

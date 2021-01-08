"""
    bin_to_leaves(lnr, X)

Returns leaf indices for each data point as well as indices of every leaf.
"""
function bin_to_leaves(lnr::IAI.OptimalTreeLearner, X::DataFrame)
    n_nodes = IAI.get_num_nodes(lnr)
    all_leaves = [];
    n,p = size(X);
    for i=1:n_nodes
        if IAI.is_leaf(lnr, i)
            append!(all_leaves, i);
        end
    end
    leaf_index = IAI.apply(lnr, X)
    return leaf_index, all_leaves
end

"""
    pwl_constraint_data(lnr::IAI.OptimalTreeLearner, vks)

Creates PWL dataset from a OptimalTreeLearner
Arguments:
    lnr: OptimalTreeLearner
    vks: headers of DataFrame X, i.e. varkeys
Returns:
    Dict[leaf_number] containing [B0, B]
"""
function pwl_constraint_data(lnr::IAI.OptimalTreeLearner, vks)

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


"""
    trust_region_data(lnr:: IAI.OptimalTreeLearner, vks)

Creates trust region from a OptimalTreeLearner
Arguments:
    lnr: OptimalTreeLearner
    vks: headers of DataFrame X, i.e. varkeys
Returns:
    Dict[leaf_number] containing [B0, B]
"""
function trust_region_data(lnr:: IAI.OptimalTreeLearner, vks)
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

""" Checks if a learner is trained. """
function check_if_trained(lnr::IAI.OptimalTreeLearner)
    try
        n_nodes = IAI.get_num_nodes(lnr);
    catch err
        if isa(err, UndefRefError)
            throw(OCTException("Trees require training before being used in constraints!"))
        else
            rethrow(err)
        end
    end
end

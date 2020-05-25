function bin_to_leaves(lnr, X)
# Returns leaf indices for each data point
# as well as indices of every leaf
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

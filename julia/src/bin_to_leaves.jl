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

function leaf_fits(lnr, all_leaves)
# Returns the linear fit weights and constants for a given learner
    constants = [IAI.get_regression_constant(lnr, leaf) for leaf in all_leaves]
    weights = [IAI.get_regression_weights(lnr, leaf) for leaf in all_leaves]
    return weights, constants
end
        
function monomials(weights, constants)
# Returns set of monomial functions for the leaves 
    return None
end
            
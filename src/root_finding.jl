#=
root_finding:
- Author: Berk
- Date: 2020-08-07
Root-finding methods sampling nonlinear functions
=#

function normalized_data(bbl::Union{BlackBoxClassifier, BlackBoxRegressor})
    """ Normalizes and returns data (0-1) by lower and upper bounds."""
    bounds = get_bounds(bbl.vars)
    vks = string.(bbl.vars)
    lbs = [minimum(val) for (key, val) in bounds]
    ubs = [maximum(val) for (key, val) in bounds]
    normalized_X = reduce(hcat,[(bbl.X[:, i] .- lbs[i]) ./(ubs[i] - lbs[i]) for i=1:length(vks)]);
    return normalized_X
end

function build_knn_tree(bbl::Union{BlackBoxClassifier, BlackBoxRegressor})
    """ Builds an efficient KNN tree over existing data."""
    X = normalized_data(bbl); #TODO: optimize calls to normalize data.
    kdtree = KDTree(X', reorder = false);
    bbl.knn_tree = kdtree;
    return
end

function find_knn(bbl::Union{BlackBoxClassifier, BlackBoxRegressor}; k::Int64 = 10)
    """ Returns idxs and Euclidian distances of KNNs of each data point. """
    idxs, dists = knn(bbl.knn_tree, bbl.knn_tree.data, k+1);
    return idxs, dists
end

""" Classifies KNN domains by feasibility. """
function classify_patches(bbc::BlackBoxClassifier, idxs::Array)
    arr = []
    for idx in idxs
        signs = [bbc.Y[i] for i in idx];
        if all(signs .>= 0)
            push!(arr, "feas")
        elseif all(signs .< 0)
            push!(arr, "infeas")
        else
            push!(arr, "mixed")
        end
    end
    return arr
end

function secant_method(X::DataFrame, Y::Array)
    """ Generates samples estimated to be near a zero from X and Y data with mixed feasibility. """
    feas = Y .>= 0;
    np = DataFrame([Float64 for i in propertynames(X)], propertynames(X))
    for i = 1:length(feas)
        for j = i+1:length(feas)
            if feas[i] + feas[j] == 1
                 if !isinf(Y[j]) && !isinf(Y[i]) # For non-infinite outputs
                    push!(np, Array(X[j,:]) - (Array(X[j,:]) - Array(X[i,:])) ./ (Y[j] - Y[i]) * Y[j])
                 else # For Infs, try midpoint.
                    push!(np, Array(X[j,:]) - (Array(X[j,:]) - Array(X[i,:])) ./ 2.)
                 end
            end
        end
    end
    return np
end
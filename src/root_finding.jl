#=
root_finding:
- Author: Berk
- Date: 2020-08-07
Root-finding methods sampling nonlinear functions
=#
""" 
    normalized_data(bbl::BlackBoxLearner)
    normalized_data(X::DataFrame)

Normalizes and returns data (0-1) by lower and upper bounds.
If used on DataFrame, returns lbs and ubs as well. 
"""
function normalized_data(bbl::BlackBoxLearner)
    bounds = get_bounds(bbl.vars)
    vks = string.(bbl.vars)
    lbs = [minimum(bounds[key]) for key in bbl.vars]
    ubs = [maximum(bounds[key]) for key in bbl.vars]
    normalized_X = reduce(hcat,[(bbl.X[:, vks[i]] .- lbs[i]) ./(ubs[i] - lbs[i]) for i=1:length(lbs)]);
    return normalized_X
end

function normalized_data(X::DataFrame)
    lbs = [minimum(col) for col in eachcol(X)]
    ubs = [maximum(col) for col in eachcol(X)]
    normalized_X = reduce(hcat,[(X[:, i] .- lbs[i]) ./ (ubs[i] - lbs[i]) for i=1:length(lbs)]);
    return normalized_X, lbs, ubs
end

function build_knn_tree(bbl::BlackBoxLearner)
    """ Builds an efficient KNN tree over existing data."""
    X = normalized_data(bbl); #TODO: optimize calls to normalize data.
    kdtree = KDTree(X', reorder = false);
    bbl.knn_tree = kdtree;
    return
end

function find_knn(bbl::BlackBoxLearner; k::Int64 = 10)
    """ Returns idxs and Euclidian distances of KNNs of each data point. """
    idxs, dists = knn(bbl.knn_tree, bbl.knn_tree.data, k+1);
    return idxs, dists
end

""" Classifies KNN domains by feasibility. """
function classify_patches(bbc::BlackBoxLearner, idxs::Array, threshold = 0)
    arr = []
    for idx in idxs
        signs = [bbc.Y[i] for i in idx];
        if all(signs .>= threshold)
            push!(arr, "feas")
        elseif all(signs .< threshold)
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
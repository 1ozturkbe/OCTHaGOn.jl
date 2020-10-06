#=
root_finding:
- Author: Berk
- Date: 2020-08-07
Root-finding methods sampling nonlinear functions
=#

function normalized_data(bbf::BlackBoxFunction)
    """ Normalizes and returns data (0-1) by lower and upper bounds."""
    bounds = get_bounds(bbf.vars)
    vks = string.(bbf.vars)
    lbs = [minimum(value) for (key, value) in bounds]
    ubs = [maximum(value) for (key, value) in bounds]
    normalized_X = reduce(hcat,[(bbf.X[:, i] .- lbs[i]) ./(ubs[i] - lbs[i]) for i=1:length(vks)]);
    return normalized_X
end

function build_knn_tree(bbf::BlackBoxFunction)
    """ Builds an efficient KNN tree over existing data."""
    X = normalized_data(bbf); #TODO: optimize calls to normalize data.
    kdtree = KDTree(X', reorder = false);
    bbf.knn_tree = kdtree;
    return
end

function find_knn(bbf::BlackBoxFunction; k::Int64 = 10)
    """ Returns idxs and Euclidian distances of KNNs of each data point. """
    idxs, dists = knn(bbf.knn_tree, bbf.knn_tree.data, k+1);
    return idxs, dists
end

function classify_patches(bbf::BlackBoxFunction, idxs::Array)
    """ Classifies KNN domains by feasibility. """
    arr = []
    for idx in idxs
        signs = [bbf.Y[i] for i in idx];
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

# function sampling_policy(bbf::BlackBoxFunction, idxs::Array)
#     class_dict = classify_patches(bbf, idxs)
#     feas_ratio = bbf.feas_ratio

function secant_method(X::DataFrame, Y::Array)
    """ Generates samples estimated to be near a zero from X and Y data with mixed feasibility. """
    feas = Y .>= 0;
    np = DataFrame([Float64 for i in propertynames(X)], propertynames(X))
    for i = 1:length(feas)
        for j = i+1:length(feas)
            if feas[i] + feas[j] == 1
                push!(np, Array(X[j,:]) - (Array(X[j,:]) - Array(X[i,:])) ./ (Y[j] - Y[i]) * Y[j]);
            end
        end
    end
    return np
end
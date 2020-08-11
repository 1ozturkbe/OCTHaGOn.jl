#=
root_finding:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-07
Root-finding methods sampling nonlinear functions
=#

using Combinatorics
using DataFrames
using Gurobi
using JuMP

include("exceptions.jl");
include("black_box_function.jl");
include("model_data.jl");

function normalized_data(bbf::Union{BlackBoxFunction, ModelData})
    """ Normalizes and returns data (0-1) by lower and upper bounds."""
    lbs_arr = [bbf.lbs[vk] for vk in bbf.vks];
    ubs_arr = [bbf.ubs[vk] for vk in bbf.vks];
    normalized_X = reduce(hcat,[(bbf.X[:, i] .- lbs_arr[i]) ./(ubs_arr[i] - lbs_arr[i]) for i=1:length(bbf.vks)]);
    return normalized_X
end

function build_knn_tree(bbf::BlackBoxFunction)
    """ Builds an efficient KNN tree over existing data."""
    X = normalized_data(bbf);
    kdtree = KDTree(X', reorder = false);
    bbf.knn_tree = kdtree;
    return
end

function find_knn(bbf::BlackBoxFunction; k::Int64 = 5)
    """ Returns idxs and Euclidian distances of KNNs of each data point. """
    idxs, dists = knn(bbf.knn_tree, bbf.knn_tree.data, k+1);
    return idxs, dists
end

function classify_patches(bbf::BlackBoxFunction, idxs::Array)
    class_dict = Dict("feas" => Array[], "mixed" => Array[], "infeas" => Array[])
    for idx in idxs
        values = [bbf.Y[i] for i in idx];
        if all(values .>= 0)
            push!(class_dict["feas"], idxs)
        elseif all(values .<= 0)
            push!(class_dict["infeas"], idxs)
        else
            push!(class_dict["mixed"], idxs)
        end
    end
    return class_dict
end

function generate_zero_samples(bbf, idx::Array)
    """ Generates samples estimated to be near a zero from a KNN patch with mixed feasibility. """
    X = Matrix(bbf.X[idx, :]);
    Y = bbf.Y[idx, :];
    feas = Y .>= 0;
    np = Array[];
    m = Model(solver=GurobiSolver())
    @variable(m, x[vk in bbf.vks])
    @variable(m, y)
    @objective(m, Min, y^2)
    OCT.add_bounds!(m, x, bbf);
    for i = 1:length(feas)
        for j = i+1:length(feas)
            if feas[i] + feas[j] >= 0
                grad = (Y[j] - Y[i]) ./ (X[j,:] - X[i,:]);
                @constraint(m, con, y >= grad*(x - X[j,:]) + X[j,:])
                solve(m)
                ns = getvalue(x);
                push!(np, ns)
                delete(model, con)
            end
        end
    end
    np = reduce(hcat, np)
end

function generate_projection_samples(bbf, idxs::Array)
    """ Generates samples using a local linear model from a KNN patch with full in/feasibility. """
end

# function grad_and_Hessian(points::AbstractArray)
#     k, n_dims = size(points);
#     @assert k >= 2*n_dims + n_dims^2/2
# end
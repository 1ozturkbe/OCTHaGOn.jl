#=
test_sampling:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-07
=#

using Test
using Combinatorics
using Gurobi
using JuMP
using LatinHypercubeSampling
using MosekTools
using MathOptInterface
using NearestNeighbors

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__

# Import a model
filename = string("../data/cblib.zib.de/shortfall_20_15.cbf.gz");
md = OCT.CBF_to_ModelData(filename);
OCT.find_bounds!(md)
OCT.update_bounds!(md, lbs = Dict(vk => 0 for vk in md.vks));
OCT.update_bounds!(md, ubs = Dict(vk => 1 for vk in md.vks));

# Pick one of the functions, and sample.
bbf = md.fns[1];
bbf.n_samples= 500;
OCT.sample_and_eval!(bbf);

# Create a NearestNeighbors Tree and compute gradients
OCT.build_knn_tree(bbf);
idxs, dists = OCT.find_knn(bbf, k=10);
@test all(i in idxs[i] for i = 1:size(bbf.X, 1));
class_dict = OCT.classify_patches(bbf, idxs);

# Check random sampling vs. optimal sampling



# combs = collect(combinations(1:k,2));
# cluster_values = Array[];
# for i = 1:length(idxs)
#     push!(cluster_values, bbf.Y[j for j in idxs[i]])
# end
# cluster_values = [bbf.Y[i for i in idx] for idx in idxs]
# # Doing a central difference for the gradient, based on KNN
# grads = Array[];
#
# for i = 1:n_samples
#     idx = filter!(x -> x != i, idxs[i]);
#     points = Matrix(bbf.X[idx, :]);
#     values = bbf.Y[idx];
#     all_grad = [(values[comb[1]] - values[comb[2]]) ./ (points[comb[1],:] - points[comb[2], :]) for comb in combs]
#     push!(grads, sum(all_grad)./length(all_grad));
# end
#
# function local_grad(points::Matrix, values::Array, combs::Union{Nothing, Array} = nothing)
# """ Uses central differencing over all point combinations to approximate the gradient"""
#     grad = [];
#     n_points, n_dims = size(points);
#     if isnothing(combs)
#         combs = collect(combinations(1:n_points,2))
#     end
#     for i in combs
#         grad
# end
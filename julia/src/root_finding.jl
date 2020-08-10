#=
root_finding:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-07
Root-finding methods sampling nonlinear functions
=#

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
    X = normalized_data(bbf);
    kdtree = KDTree(X');
    bbf.knn_tree = kdtree;
    return
end

# function grad_and_Hessian(points::AbstractArray)
#     k, n_dims = size(points);
#     @assert k >= 2*n_dims + n_dims^2/2
# end
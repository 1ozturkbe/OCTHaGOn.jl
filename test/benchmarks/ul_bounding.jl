using LinearAlgebra
using StatsBase
using Plots

# m = minlp(false)
# optimize!(m)
# mcost = JuMP.getobjectivevalue(m)
# msol = solution(m)

gm = test_gqp()
set_optimizer(gm, CPLEX_SILENT)

# bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
bbr = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor][1]
uniform_sample_and_eval!(gm)
scatter(bbr.X[:,"x[1]"], bbr.X[:, "x[2]"], bbr.Y)


# # Learning BBCs just for fun. 
# for bbc in bbcs
#     learn_constraint!(bbc)
#     update_tree_constraints!(gm, bbc)
# end

# # Different types of regression learners for testing purposes. 
learn_constraint!(bbr, regression_sparsity=0, max_depth = 3, minbucket = 2*length(bbr.vars) + 2)
update_tree_constraints!(gm, bbr)
show_trees(bbr)
optimize!(gm)

# Plotting all of the different leaves
scatter()
idx = length(bbr.learners)
ul_data = bbr.ul_data[idx]
lnr = bbr.learners[idx]
leaf_binnings = IAI.apply(lnr, bbr.X)
for leaf in find_leaves(lnr)
    in_leaf = findall(x -> x == leaf, leaf_binnings)
    Plots.display(scatter!(bbr.X[in_leaf,"x[1]"], bbr.X[in_leaf, "x[2]"], bbr.Y[in_leaf], label = leaf))
end

# # Scoring different trees
# for tree in bbr.learners
#     println("Tree score: ", IAI.score(tree, bbr.X, bbr.Y))
# end

# # Sample in all leaves that contain the actual solution, and retrain. 
# # How about minimizing gradient cross products? 

# # Using local KNN regression
build_knn_tree(bbr);
idxs, dists = find_knn(bbr, k=length(bbr.vars) + 1);
X = Matrix(bbr.X)
bbr_gradients = evaluate_gradient(bbr, bbr.X);
up_idxs = []
down_idxs = []
mixed_idxs = []
for i = 1:size(bbr.X, 1)
    diffs = [X[i, :] - X[j, :] for j in idxs[i]]
    center_grad = bbr_gradients[i]
    under_offsets = [-dot(center_grad, differ) for differ in diffs]
    actual_offsets = bbr.Y[idxs[i]] .- bbr.Y[i]
    if all(actual_offsets .>= under_offsets .- 1e-10)
        append!(up_idxs, i)
    elseif all(actual_offsets .- 1e-10 .<= under_offsets )
        append!(down_idxs, i)
    else
        append!(mixed_idxs, i)
    end
end


# DETECTING CONVEXITY



all_leaves = find_leaves(lnr)
points_in_leaves = Dict(leaf => [] for leaf in all_leaves)
for i=1:length(leaf_binnings)
    push!(points_in_leaves[leaf_binnings[i]], i)
end
centroids = Dict(leaf => [mean(col) for col in eachcol(
             bbr.X[points_in_leaves[leaf], :])] for leaf in all_leaves)

convex_bins = Any[[leaf] for leaf in all_leaves]
parents = unique([IAI.get_parent(lnr, leaf[1]) for leaf in convex_bins])

for par in parents
    lower_child = IAI.get_lower_child(lnr, par)
    upper_child = IAI.get_upper_child(lnr, par)
    if IAI.is_leaf(lnr, lower_child) && IAI.is_leaf(lnr, upper_child)
        α0, α = ul_data[lower_child]
        β0, β = ul_data[upper_child]
        lower_X = bbr.X[points_in_leaves[lower_child], :]
        lower_Y = bbr.Y[points_in_leaves[lower_child]]
        lower_predictions = Matrix(lower_X)*α .+ α0
        upper_X = bbr.X[points_in_leaves[upper_child], :]
        upper_Y = bbr.Y[points_in_leaves[upper_child]]
        upper_predictions = Matrix(upper_X)*β .+ β0

        l_in_l = lower_Y - Matrix(lower_X)*α .- α0
        u_in_u = upper_Y - Matrix(upper_X)*β .- β0
        l_in_u = lower_Y - Matrix(lower_X)*β .- β0
        u_in_l = upper_Y - Matrix(upper_X)*α .- α0
        if all(l_in_u .>= 0) && all(u_in_l .>= 0)
            push!(convex_bins, [par, [lower_child, upper_child]])

        else
            println("Violations ", (sum(l_in_u .>= 0) + sum(u_in_l .>= 0))/(length(l_in_u) + length(u_in_l)))
        end
    end
end




        # parents = [all_leaves[i]]
        # while IAI.get_depth(lnr, parents[end]) > 0
        #     append!(parents, IAI.get_parent(lnr, parents[end]))
        # end
        # upperDict[all_leaves[i]] = []
        # lowerDict[all_leaves[i]] = []
        # for j in parents[2:end]
        #     # For each parent, define trust region with binary variables
        #     threshold = IAI.get_split_threshold(lnr, j)
        #     if IAI.is_hyperplane_split(lnr, j)
        #         weights = IAI.get_split_weights(lnr, j)
        #         weights = weights[1]
        #     else
        #         feature = IAI.get_split_feature(lnr, j)
        #         weights = Dict(feature => 1)
        #     end
        #     upper = IAI.get_upper_child(lnr, j) in parents # Checking upper vs. lower split
        #     α = []
        #     for i = 1:size(vks, 1)
        #         if vks[i] in keys(weights)
        #             append!(α, weights[vks[i]])
        #         else
        #             append!(α, 0.0)
        #         end
        #     end
        #     if upper
        #         append!(upperDict[all_leaves[i]], [[threshold, α]])
        #     else
        #         append!(lowerDict[all_leaves[i]], [[threshold, α]])


"""
    knn_outward_from_leaf(bbl::BlackBoxLearner, leaf_in::Int64 = find_leaf_of_soln(bbl))

Binary knn samples from current leaf to surrounding leaves.   
"""
function knn_outward_from_leaf(bbl::BlackBoxLearner, leaf_in::Int64 = find_leaf_of_soln(bbl), k = 10)
    leaves = IAI.apply(bbl.learners[end], bbl.X);
    leaf_neighbors = findall(x -> x == leaf_in, leaves);
    df = DataFrame([Float64 for i in string.(bbl.vars)], string.(bbl.vars))
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k = k);
    leaf_neighbors = findall(x -> x == leaf_in, leaves)
    leaf_neighbors = leaf_neighbors[randperm(MersenneTwister(1234), length(leaf_neighbors))]; # Shuffling pre while loop
    ct = 1
    while size(df, 1) <= get_param(bbl, :n_samples)/2 && ct <= length(leaf_neighbors)
        neighbor = leaf_neighbors[ct] # sampling from inside outward
        stranger_idxs = findall(i -> leaves[i] != leaf_in, idxs[neighbor])
        for idx in stranger_idxs
            dist = dists[neighbor][idx]
            if dist >= get_param(gm, :tighttol)
                np = (Array(bbl.X[neighbor, :]) + Array(bbl.X[idxs[neighbor][idx], :]))./2
                push!(df, np)
            end
        end
        ct += 1
    end
    return df
end

# function knn_around_lastsol(bbl::BlackBoxLearner)
#     lastsol = getvalue.(bbl.vars)
#     build_knn_tree(bbl)
#     pts = NearestNeighbors.inrange(bbl.knn_tree, lastsol, 0.05)

# Initializing, and solving via Ipopt
m = random_qp(5, 3, 2)
optimize!(m)
mcost = JuMP.getobjectivevalue(m)
msol = getvalue.(m[:x])

# Trying thresholding method 
# using StatsBase
gm = gmify_random_qp(m)
bbl = gm.bbls[1]
set_param(bbl, :n_samples, 500)
uniform_sample_and_eval!(gm)
uppers = [20.]
upper_learners = []
upper_ul_data = []
lowers = [0.]
lower_learners = []
lower_ul_data = []
actuals = []
estimates = []
tightnesses = []
errors = []

kwargs = Dict()
lnr = base_classifier()
M = 1e5
push!(upper_learners, learn_from_data!(bbl.X, bbl.Y .<= uppers[end], IAI.clone(lnr); fit_classifier_kwargs(; kwargs...)...))
push!(upper_ul_data, ul_boundify(upper_learners[end], bbl.X, bbl.Y))
push!(lower_learners, learn_from_data!(bbl.X, bbl.Y .>= lowers[end], IAI.clone(lnr); fit_classifier_kwargs(; kwargs...)...))
push!(lower_ul_data, ul_boundify(lower_learners[end], bbl.X, bbl.Y))

mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbl.vars, upper_learners[end]; M=1e5)
feas_leaves = collect(keys(leaf_variables))
for leaf in feas_leaves
    (α0, α), (β0, β), (γ0, γ) = upper_ul_data[end][leaf]
    push!(mi_constraints[leaf], @constraint(gm.model, bbl.dependent_var <= α0 + sum(α .* bbl.vars) + M * (1 .- leaf_variables[leaf])))   
end
merge!(append!, bbl.mi_constraints, mi_constraints)
merge!(append!, bbl.leaf_variables, leaf_variables) # This is problematic...

mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbl.vars, lower_learners[end]; M=1e5)
feas_leaves = collect(keys(leaf_variables))
for leaf in feas_leaves
    (α0, α), (β0, β), (γ0, γ) = lower_ul_data[end][leaf]
    push!(mi_constraints[leaf], @constraint(gm.model, bbl.dependent_var + M * (1 .- leaf_variables[leaf]) >= β0 + sum(β .* bbl.vars)))
end
merge!(append!, bbl.mi_constraints, mi_constraints)
merge!(append!, bbl.leaf_variables, leaf_variables)

# Getting solution
optimize!(gm)
push!(estimates, getvalue(bbl.dependent_var))
push!(actuals, bbl(solution(gm))[1])
clear_tree_constraints!(gm)

# Updating bounds
if actuals[end] <= lowers[end]
    push!(uppers, lowers[end])
    push!(lowers, minimum(bbl.Y))
elseif lowers[end] <= actuals[end] <= uppers[end]
    push!(uppers, actuals[end])
    push!(lowers, (uppers[end] + lowers[end])/2) # Binary reduce
elseif actuals[end] >= uppers[end]
    push!(uppers, actuals[end])
    push!(lowers, lowers[end])
end

# Adding all gradient constraints in leaf, just in case





# threshold = quantile(bbl.Y, 0.1)
# push!(thresholds, threshold)

# while length(bbl.learners) <= 5
#     learn_constraint!(gm, threshold=thresholds[end])
#     add_tree_constraints!(gm)
#     optimize!(gm)

#     leaf_in = find_leaf_of_soln(bbl)
#     #UL_data for the leaf
#     (α0, α), (β0, β), (γ0, γ) = bbl.ul_data[end][leaf_in]
#     push!(uppers, α0 + sum(α .* getvalue.(bbl.vars)))
#     push!(lowers, β0 + sum(β .* getvalue.(bbl.vars)))
#     push!(actuals, bbl(solution(gm))[1])
#     push!(estimates, JuMP.getobjectivevalue(gm.model))
#     push!(errors, abs((estimates[end] - actuals[end]) ./ (maximum(bbl.Y) - minimum(bbl.Y))))
#     # Resample
#     # df = knn_outward_from_leaf(bbl, leaf_in)
#     # eval!(bbl, df)

#     # Do something about uppers, lowers and actuals. 
#     # push!(thresholds, actuals[end])
#     # if actuals[end] <= lowers[end]
#     #     push!(thresholds, lowers[end])
#     # elseif lowers[end] <= actuals[end] <= uppers[end]
#     #     push!(thresholds, (uppers[end] - actuals[end])/2 + actuals[end])
#     # elseif actuals[end] >= uppers[end]
#     #     push!(thresholds, actuals[end])
#     #     push!(bbl.learners, bbl.learners[end])
#     #     push!(bbl.ul_data, ul_boundify(bbl.learners[end], bbl.X, bbl.Y))
#     # end

#     # Find a way to add a cut and remove certain data. 

#     if lowers[end] <= actuals[end] <= estimates[end] <= uppers[end]
#         push!(thresholds, (actuals[end] + estimates[end])/2)
#     elseif lowers[end] <= estimates[end] <= actuals[end] <= uppers[end]
#         push!(thresholds, actuals[end])
#     elseif actuals[end] >= uppers[end]
#         push!(thresholds, actuals[end])
#         push!(bbl.learners, bbl.learners[end])
#         push!(bbl.ul_data, ul_boundify(bbl.learners[end], bbl.X, bbl.Y))
#     end
#     clear_tree_constraints!(gm, bbl)
# end

# When doing threshold training, make sure I can ignore data above. 

# Plotting
using Plots
plot(lowers[1:5], label = "lowers")
plot!(actuals[1:5], label = "actuals")
plot!(estimates[1:5], label = "estimates")
plot!(uppers[1:5], label = "uppers", thickness_scaling = 2)

# What if I just used cutting planes? 

clear_data!(gm)
uniform_sample_and_eval!(gm)
gradvals = evaluate_gradient(bbl, bbl.X)

# Testing adding gradient cuts
for i=1:size(bbl.X, 1)
    @constraint(gm.model, bbl.dependent_var >= sum(gradvals[i] .* (bbl.vars .- Array(bbl.X[i, :]))) + bbl.Y[i])
end
optimize!(gm)
@test all(isapprox(Array(solution(gm))[i], [0.5, 1.0, 11.25][i], atol=0.1) for i=1:3)

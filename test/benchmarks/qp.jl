using LatinHypercubeSampling

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

""" Solves the QP with cutting planes only. """
function cutting_plane_solve(gm::GlobalModel)
    bbr = gm.bbls[1]
    gradvals = evaluate_gradient(bbr, bbr.X)
    # Testing adding gradient cuts
    for i=1:size(bbr.X, 1)
        @constraint(gm.model, bbr.dependent_var >= sum(gradvals[i] .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i])
    end
    optimize!(gm)   
end

""" Very coarse solution for gm to check for feasibility and an initial starting point. """
function surveysolve(gm::GlobalModel)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    for bbc in bbcs
        learn_constraint!(bbc)
        update_tree_constraints!(gm, bbc)
    end
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    bbr = bbrs[1]
    learn_constraint!(bbr, regression_sparsity = 0, max_depth = 2)
    update_tree_constraints!(gm, bbr)
    optimize!(gm)
    return
end

function refine_thresholds(gm::GlobalModel, bbr::BlackBoxRegressor)
    if length(bbr.active_trees) == 1
        best_lower = getvalue(bbr.dependent_var)
        best_upper = bbr.Y[end]
        learn_constraint!(bbr, threshold = "upper" => best_upper)
        update_tree_constraints!(gm, bbr)
        learn_constraint!(bbr, threshold = "lower" =>best_lower)
        update_tree_constraints!(gm, bbr)
        return
    elseif length(bbr.active_trees) == 2
        bds = Dict(collect(values(bbr.active_trees))) # TODO: have a cleaner system for this. 
        old_lower = bds["lower"]
        old_upper = bds["upper"]
        new_lower = getvalue(bbr.dependent_var)
        new_upper = bbr.Y[end]
        if new_upper <= old_upper # tightening the upper bound
            learn_constraint!(bbr, threshold = "upper" => new_upper)
            update_tree_constraints!(gm, bbr)
            learn_constraint!(bbr, # binary reduce the lower bound
                threshold = "lower" => (maximum([new_lower, old_lower]) + new_upper)/2)
        end
        if new_lower > old_lower
            learn_constraint!(bbr, threshold = "lower" => new_lower)
            update_tree_constraints!(gm, bbr)
        else
            learn_constraint!(bbr, threshold = "lower" => new_lower) # TODO: determine if better way exists
        end 
        return 
    else 
        throw(OCTException("Cannot refine $(bbr.name) thresholds without having solved " *
                           "GlobalModel $(gm.name) with valid approximations first." ))
    end
end

"""
    leaf_sample(bbl::BlackBoxLearner)

Gets Latin Hypercube samples that fall in the leaf of the last solution.
"""

function leaf_sample(bbc::BlackBoxClassifier)
    # TODO: write. 
    return
end

function leaf_sample(bbr::BlackBoxRegressor)
    length(bbr.active_trees) == 2 || throw(OCTException("Can only leaf sample BBRs with upper/lower " *
                                                        "classifiers."))
    upper_leaf, lower_leaf = sort(find_leaf_of_soln(bbr))
    upper_leafneighbor = [];
    lower_leafneighbor = [];
    for (tree_idx, threshold) in bbr.active_trees
        if threshold.first == "upper"
            append!(upper_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== -upper_leaf)
        elseif threshold.first == "lower"
            append!(lower_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== lower_leaf)
        end
    end
    idxs =  findall(x -> x .>= 0.5, upper_leafneighbor .* lower_leafneighbor)
    lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
    ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
    plan, _ = LHCoptim(get_param(bbr, :n_samples), length(bbr.vars), 3);
    X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
   return DataFrame(X, string.(bbr.vars))
end


# Initializing, and solving via Ipopt
m = random_qp(5, 3, 4)
optimize!(m)
mcost = JuMP.getobjectivevalue(m)
msol = getvalue.(m[:x])

# Trying thresholding method 
# using StatsBase
gm = gmify_random_qp(m)
bbr = gm.bbls[1]
uniform_sample_and_eval!(gm)
surveysolve(gm)
refine_thresholds(gm, bbr)
optimize!(gm)

abstol = 1e-1
old_lower = bds["lower"]
old_upper = bds["upper"]
iterations = 0
while old_upper - old_lower >= abstol && iterations <= 3
    df = leaf_sample(bbr);
    eval!(bbr, df);
    refine_thresholds(gm, bbr);
    optimize!(gm);
    bds = Dict(collect(values(bbr.active_trees))); # TODO: have a cleaner system for this. 
    old_lower = bds["lower"];
    old_upper = bds["upper"];
    println("Next bounds: " * string([old_lower, old_upper]))
    global iterations += 1;
end


# Getting solution

# Updating bounds
# if actuals[end] <= lowers[end]
#     push!(uppers, lowers[end])
#     push!(lowers, minimum(bbl.Y))
# elseif lowers[end] <= actuals[end] <= uppers[end]
#     push!(uppers, actuals[end])
#     push!(lowers, (uppers[end] + lowers[end])/2) # Binary reduce
# elseif actuals[end] >= uppers[end]
#     push!(uppers, actuals[end])
#     push!(lowers, lowers[end])
# end

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
# using Plots
# plot(lowers[1:5], label = "lowers")
# plot!(actuals[1:5], label = "actuals")
# plot!(estimates[1:5], label = "estimates")
# plot!(uppers[1:5], label = "uppers", thickness_scaling = 2)
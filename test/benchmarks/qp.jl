using LatinHypercubeSampling
using Plots

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

function refine_thresholds(gm::GlobalModel, bbr::BlackBoxRegressor)
    if length(bbr.active_trees) == 1
        best_lower = getvalue(bbr.dependent_var)
        best_upper = bbr(solution(gm))[1]
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
        new_upper = bbr(solution(gm))[1]
        # Updating upper bounds
        if new_upper <= old_upper
            learn_constraint!(bbr, threshold = "upper" => new_upper)
            update_tree_constraints!(gm, bbr)
        else
            learn_constraint!(bbr, threshold = "upper" => old_upper) #TODO add warmstarts here. 
            update_tree_constraints!(gm, bbr)
        end
        # Updating lower bounds
        if new_lower >= new_upper
            learn_constraint!(bbr, threshold = "lower" => old_lower)
            update_tree_constraints!(gm, bbr)
            return 
        elseif old_lower <= new_lower # tightening the lower bound
            learn_constraint!(bbr, # binary reduce the lower bound
                threshold = "lower" => (maximum([new_lower, old_lower]) + new_upper)/2)
            update_tree_constraints!(gm, bbr)
            return
        return 
        end
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
    last_leaf = find_leaf_of_soln(bbc)
    idxs = IAI.apply
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
    if length(idxs) == 0
        @warn("No points in $(bbr.name) in the intersection of trees. Widening sampling. ")
        idxs =  findall(x -> x .>= 0.5, upper_leafneighbor + lower_leafneighbor)
    end
    lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
    ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
    plan, _ = LHCoptim(get_param(bbr, :n_samples), length(bbr.vars), 3);
    X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
    return DataFrame(X, string.(bbr.vars))
end


# Initializing, and solving via Ipopt
Random.seed!(1212)
m = random_qp(5,3,4)
# m = random_qp(2,1,2)

optimize!(m)
mcost = JuMP.getobjectivevalue(m)
msol = getvalue.(m[:x])

# Trying thresholding method 
gm = gmify_random_qp(m)
bbr = gm.bbls[1]
uniform_sample_and_eval!(gm)
surveysolve(gm)
refine_thresholds(gm, bbr)
optimize!(gm)

bds = Dict(collect(values(bbr.active_trees))); # TODO: have a cleaner system for this. 
old_lower = bds["lower"]
old_upper = bds["upper"]

iterations = 0
reltol = 1e-5
set_param(gm, :reltol, reltol)
while abs(bbr(solution(gm))[1] - bbr(DataFrame(gm.solution_history[end-1, :]))[1]) >= reltol*bbr(solution(gm))[1] && 
        iterations <= 5
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
progress = [bbr(DataFrame(gm.solution_history[i, :]))[1] for i = 1:size(gm.solution_history,1)]

# Plotting
# using Plots
# plot(lowers[1:5], label = "lowers")
# plot!(actuals[1:5], label = "actuals")
# plot!(estimates[1:5], label = "estimates")
# plot!(uppers[1:5], label = "uppers", thickness_scaling = 2)
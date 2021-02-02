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
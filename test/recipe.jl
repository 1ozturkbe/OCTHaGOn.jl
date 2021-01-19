# """ Very coarse solution for gm to check for feasibility and an initial starting point. """
# function surveysolve(gm::GlobalModel)
#     bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
#     learn_constraint!(bbcs)
#     bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
#     bbr = bbrs[1]
#     learn_constraint!(bbr, regression_sparsity = 0, max_depth = 2)
#     ul_data = ul_boundify(bbr.learners[end], bbr.X, bbr.Y)
#     # Setting an extremely conservative upper bound for the objective. 
#     # l_con = @constraint(gm.model, gm.model[:obj] >= α0 + sum(α .* bbr.vars))
#     for bbc in bbcs
#         add_tree_constraints!(gm, bbc)
#     end
#     add_tree_constraints!(gm, bbr)
#     optimize!(gm)
#     return
# end

# Set threshold using previous solution, and continue
using NearestNeighbors
using JLD
gm = minlp(true)
set_optimizer(gm, CPLEX_SILENT)
uniform_sample_and_eval!(gm)
bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
bbr = gm.bbls[findall(bbl -> bbl isa BlackBoxRegressor, gm.bbls)][1]

# learn_constraint!(bbcs)
# learn_constraint!(bbr, regression_sparsity = 0)
load_fit(gm)

add_tree_constraints!(gm)
optimize!(gm)

function check_progress(gm)
    feasibilities = evaluate_feasibility(gm) # which adds data at last solution
    last_sol = getvalue(gm.model[:obj])
    ub = bbr.Y[end]
    accuracies = OCT.evaluate_accuracy(gm)
    # last_lnr = bbr.learners[end]
    # leaf = IAI.apply(last_lnr, DataFrame(bbr.X[end, :]))[1]
    # α0, α = bbr.ul_data[end][leaf][1]
    # upper_bound = α0 + sum(α .* Array(bbr.X[end, :]))
    return last_sol, ub, feasibilities, accuracies
end

last_sol, ub, feasibilities = check_progress(gm)

clear_tree_constraints!(gm, bbr)
subthres_idx = findall(y -> y <= threshold, bbr.Y)
@assert length(subthres_idx) >= 0.05*length(bbr.Y) # TODO: or sample more. 

learn_constraint!(bbr, threshold=threshold)
add_tree_constraints!(gm, bbr)
optimize!(gm)
threshold, feasibilities, accuracies = check_progress(gm)

if !all(feasibility)
    # TODO devise new sampling/training scheme? 
end


# last_regr = bbr.learners[end]
# df = DataFrame([Float64 for i in 1:length(bbr.vars)], string.(bbr.vars))
# subthres_idx = findall(y -> y <= threshold, bbr.Y)
# subthres_leaves = IAI.apply(last_regr, bbr.X[subthres_idx,:])
# bbr.knn_tree = KDTree(OCT.normalized_data(bbr)', reorder = false)
# for leaf in unique(subthres_leaves)
#     idx_in_leaf = [i for i = 1:length(subthres_idx) if subthres_leaves[i] == leaf]
#     maxes = maximum.(eachcol(bbr.X[idx_in_leaf,:]))
#     mins = minimum.(eachcol(bbr.X[idx_in_leaf,:]))

#     idx_just_out = [] 
#     for i in idxs_in_leaf
        
#     = unique(hcat(idxs[i]), for i in idx_in_leaf)



bbr.knn_tree = KDTree(OCT.normalized_data(bbr)', reorder = false)
idxs, dists = find_knn(bbr, k=length(bbr.vars));
# feas_class = classify_patches(bbr, idxs, threshold);

for center_node in subthres_idx
    if feas_class[center_node] == "mixed"
        nodes = [idx for idx in idxs[center_node] if bbr.Y[idx] >= threshold];
        push!(nodes, center_node)
        np = secant_method(bbr.X[nodes, :], bbr.Y[nodes, :])
        append!(df, np);
    end
end

X = bbr.X[subthres_idx, :]
Y = bbr.Y[subthres_idx]
bbl.knn_tree = kdtree;

function outer_knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10)
    vks = string.(bbl.vars)
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    positives = findall(x -> x .>= 0 , bbl.Y);
    feas_class = classify_patches(bbl, idxs);
    for center_node in positives # This loop is for making sure that every possible root is sampled only once.
        if feas_class[center_node] == "mixed"
            nodes = [idx for idx in idxs[center_node] if bbl.Y[idx] < 0];
            push!(nodes, center_node)
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np);
        end
    end
    return df
end



# function neldermead_sample(bbr:BlackBoxRegressor, X::DataFrame, Y::Array, threshold; k::Int64 = length(bbr.vars))
#     vks = string.(bbl.vars)
#     df = DataFrame([Float64 for i in vks], vks)
#     build_knn_tree(bbl);
#     idxs, dists = find_knn(bbl, k=k);
#     positives = findall(x -> x .>= 0 , bbl.Y);
#     feas_class = classify_patches(bbl, idxs);
#     for center_node in positives # This loop is for making sure that every possible root is sampled only once.
#         if feas_class[center_node] == "mixed"
#             nodes = [idx for idx in idxs[center_node] if bbl.Y[idx] < 0];
#             push!(nodes, center_node)
#             np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
#             append!(df, np);
#         end
#     end
#     return df
# end

learn_constraint!(bbr, threshold=threshold)
clear_tree_constraints!(gm, bbr)
add_tree_constraints!(gm, bbr)
last_learner = bbr.learners[end]



lnr = learn_from_data!(bbr.X, bbr.Y .<= threshold, lnr::IAI.OptimalTreeLearner)

bbr.mi_constraints, bbr.leaf_variables = add_feas_constraints!(gm.model, bbr.vars, lnr;
                                            M=M, equality = bbr.equality);





lnr = base_classifier()
IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
threshold = quantile(bbr.Y, 1-interval)
nl = learn_from_data!(bbr.X, bbr.Y .<= threshold, lnr; fit_classifier_kwargs(; kwargs...)...) 
all_leaves = find_leaves(nl)
feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
leaf_idx = IAI.apply(nl, bbr.X)
upper, lower = trust_region_data(lnr, Symbol.(bbr.vars))
for leaf in feas_leaves
    idx = findall(x -> x == leaf, leaf_idx)
    Y_ub = maximum(bbr.Y[idx])
    Y_lb = minimum(bbr.Y[idx])
    β0, β = ridge_regress(X[idx,:], Y[idx])
    predictions = β0 .+ Matrix(X[idx, :]) * β
    mse_ridge = sum(((predictions - Y[idx])./Y[idx]).^2)/length(idx)
    mse_tree = sum(((IAI.predict(bbr.learners[end], bbr.X[idx,:]) - Y[idx])./Y[idx]).^2)/length(idx)
    println(IAI.score(bbr.learners[end], bbr.X[idx, :], bbr.Y[idx]))
end

total_mse = sum(((IAI.predict(bbr.learners[end], bbr.X) - bbr.Y)./bbr.Y).^2)/length(bbr.Y)

        
push!(bbr.learners, nl);
bbr.predictions = IAI.predict(nl, bbr.X)
push!(bbr.learner_kwargs, Dict(kwargs))
gm = minlp(true)
set_optimizer(gm, CPLEX_SILENT)

bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
bbr = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor][1]
uniform_sample_and_eval!(gm)

# Greedy approach
# for bbc in bbcs
#     learn_constraint!(bbc)
#     update_tree_constraints!(gm, bbc)
# end
# learn_constraint!(bbr, hyperplane_sparsity = 1, regression_sparsity=0, max_depth = 6, minbucket = 2*length(bbr.vars))
# update_tree_constraints!(gm, bbr)
# optimize!(gm)

# Sample in all leaves that contain the actual solution, and retrain. 
# How about minimizing gradient cross products? 

# Using local KNN regression
build_knn_tree(bbr);
idxs, dists = find_knn(bbr, k=length(bbr.vars) + 1);
bbr_gradients = evaluate_gradient(bbr, bbr.X)
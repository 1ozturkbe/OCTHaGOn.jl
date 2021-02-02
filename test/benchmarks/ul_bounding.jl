gm = minlp(true)
set_optimizer(gm, CPLEX_SILENT)

bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
bbr = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor][1]

# Greedy approach
for bbc in bbcs
    learn_constraint!(bbc)
    update_tree_constraints!(gm, bbc)
end
learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6, minbucket = 2*length(bbr.vars))
update_tree_constraints!(gm, bbr)
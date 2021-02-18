""" Very coarse solution for gm to check for feasibility and an initial starting point. """
function surveysolve(gm::GlobalModel)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    if !isempty(bbcs)
        for bbc in bbcs
            learn_constraint!(bbc)
            update_tree_constraints!(gm, bbc)
        end
    end
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        for bbr in bbrs
        learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6, minbucket = 3*length(bbr.vars))
        update_tree_constraints!(gm, bbr)
        end
    end
    optimize!(gm)
    return
end
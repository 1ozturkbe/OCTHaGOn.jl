""" Returns the baseline OptimalTreeRegressor. """
function base_otr()
    return IAI.OptimalTreeRegressor(
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        # Modifiables, with defaults
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6, 
        minbucket = 0.01,
        fast_num_support_restarts = 5, 
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5, 
    )
end

"""
Returns the baseline OptimalTreeClassifier, with all modifyable args. 
"""
function base_otc()
    lnr = IAI.OptimalTreeClassifier(
        hyperplane_config = (sparsity = :all,),
        random_seed = 1,
        max_depth = 5, 
        cp = 1e-6,
        minbucket = 0.01,
        fast_num_support_restarts = 5,
        localsearch = true,
        ls_num_tree_restarts = 10, 
        ls_num_hyper_restarts =  5, 
    )
end

function base_grid(lnr)
    grid = IAI.GridSearch(lnr, Dict(:criterion => [:entropy, :misclassification],
    :normalize_X => [true],
    :max_depth => [3, 5],
    :minbucket => [0.03, 0.05]))
    return grid
end

""" Turns IAI.Learners into IAI.GridSearches."""
function gridify(lnr::IAI.Learner)
    if !hasproperty(lnr, :lnr)
        grid = IAI.GridSearch(lnr);
    else
        grid = lnr;
    end
    return grid
end
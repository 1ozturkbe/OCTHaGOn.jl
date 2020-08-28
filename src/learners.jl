#=
learners:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
=#

function base_otr()
    """ Returns the baseline OptimalTreeRegressor. """
    return IAI.OptimalTreeRegressor(
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6,
        minbucket = 0.03,
        regression_sparsity = :all,
        fast_num_support_restarts = 5,
        hyperplane_config = (sparsity = :all,),
        regression_lambda = 0.00001,
        regression_weighted_betas = true,
    )
end

function base_otc()
    """ Returns the baseline OptimalTreeClassifier. """
    return IAI.OptimalTreeClassifier(
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6,
        minbucket = 0.03,
        fast_num_support_restarts = 5,
        hyperplane_config = (sparsity = :all,),
    )
end

function base_grid(lnr)
    grid = IAI.GridSearch(lnr, Dict(:criterion => [:entropy, :misclassification],
    :normalize_X => [true],
    :max_depth => [3, 5],
    :minbucket => [0.03, 0.05]))
    return grid
end

function gridify(lnr::IAI.Learner)
    """ Turns IAI.Learners into IAI.GridSearches."""
    if !hasproperty(lnr, :lnr)
        grid = IAI.GridSearch(lnr);
    else
        grid = lnr;
    end
    return grid
end
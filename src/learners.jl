#=
learners:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
=#

""" Returns the baseline OptimalTreeRegressor. """
function base_otr(max_depth::Int64 = 5, minbucket::Float64 = 0.01)
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

"""
    base_otc(max_depth::Int64 = 5, minbucket::Float64 = 0.01, localsearch::Bool = false)

Returns the baseline OptimalTreeClassifier, with parameters for different training steps.
"""

function base_otc(max_depth::Int64 = 5, minbucket::Float64 = 0.01, localsearch::Bool = false)
    return IAI.OptimalTreeClassifier(
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6,
        minbucket = minbucket,
        fast_num_support_restarts = 5,
        hyperplane_config = (sparsity = :all,),
        localsearch = localsearch,
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
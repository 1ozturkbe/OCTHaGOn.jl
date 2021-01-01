""" Returns the baseline OptimalTreeRegressor. """
function base_otr(; max_depth::Int64 = 5, minbucket::Float64 = 0.01)
    return IAI.OptimalTreeRegressor(
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        # Modifiables
        random_seed = get(kwargs, :random_seed, 1),
        max_depth = get(kwargs, :max_depth, 5),
        cp = get(kwargs, :cp, 1e-6),
        minbucket = get(kwargs, :minbucket, 0.01),
        fast_num_support_restarts = get(kwargs, :fast_num_support_restarts, 5),
        regression_lambda = get(kwargs, :regression_lambda, 1e-5),
        ls_num_tree_restarts = get(kwargs, :ls_num_tree_restarts, 10),
        ls_num_hyper_restarts =  get(kwargs, :ls_num_hyper_restarts, 5),
    )
end

"""
    base_otc(;kwargs...)

Returns the baseline OptimalTreeClassifier, with parameters for different training steps.
"""

function ase_otc(; kwargs...)
    lnr = IAI.OptimalTreeClassifier(
        hyperplane_config = (sparsity = :all,),
        # Modifiables
        random_seed = get(kwargs, :random_seed, 1),
        max_depth = get(kwargs, :max_depth, 5),
        cp = get(kwargs, :cp, 1e-6),
        minbucket = get(kwargs, :minbucket, 0.01),
        fast_num_support_restarts = get(kwargs, :fast_num_support_restarts, 5),
        localsearch = get(kwargs, :localsearch, true),
        ls_num_tree_restarts = get(kwargs, :ls_num_tree_restarts, 10),
        ls_num_hyper_restarts =  get(kwargs, :ls_num_hyper_restarts, 5),
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
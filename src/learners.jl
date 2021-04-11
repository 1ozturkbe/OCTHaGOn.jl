""" Returns the baseline OptimalTreeLearners, with all valid training args. """
function base_classifier()
    IAI.OptimalTreeClassifier(
        hyperplane_config = (sparsity = :all,),
        random_seed = 1,
        max_depth = 5, 
        cp = 1e-6,
        minbucket = 0.01,
        fast_num_support_restarts = 5,
        localsearch = true,
        ls_num_tree_restarts = 10, 
        ls_num_hyper_restarts =  5)
end

function base_regressor()
    IAI.OptimalTreeRegressor(
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end

function base_rf_regressor()
    IAI.Heuristics.RandomForestRegressor(
        num_trees = 20,
        hyperplane_config = (sparsity = :all,),
        regression_sparsity = :all,
        regression_weighted_betas = true,
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        regression_lambda = 1e-5,
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end

function base_rf_classifier()
    IAI.Heuristics.RandomForestClassifier(
        num_trees = 20,
        hyperplane_config = (sparsity = :all,),
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6, 
        minbucket = 0.02,
        fast_num_support_restarts = 5, 
        ls_num_tree_restarts = 10,
        ls_num_hyper_restarts =  5)
end
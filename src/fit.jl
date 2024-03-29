""" Helper function for merging learner arguments. """
function merge_kwargs(valid_keys; kwargs...)
    nkwargs = Dict{Symbol, Any}() 
    for item in kwargs
        if item.first in valid_keys
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! 
TODO: Add nkwargs that can be changed! """
function fit_regressor_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}()
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! 
TODO: Add nkwargs that can be changed! """
function fit_classifier_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}(:sample_weight => :autobalance)
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Function that preprocesses merging of kwargs for IAI.OptimalTreeClassifier, to avoid errors! """
function classifier_kwargs(; kwargs...)
    valid_keys = [:random_seed, :max_depth, :cp, :minbucket, 
                  :fast_num_support_restarts, :localsearch, 
                  :ls_num_hyper_restarts, :ls_num_tree_restarts, 
                  :hyperplane_config, :criterion]
    merge_kwargs(valid_keys; kwargs...)
end

""" Function that preprocesses merging of kwargs for IAI.OptimalTreeRegressor, to avoid errors! """
function regressor_kwargs(; kwargs...)
    valid_keys = [:random_seed, :max_depth, :cp, :minbucket, 
                  :fast_num_support_restarts, :localsearch, 
                  :ls_num_hyper_restarts, :ls_num_tree_restarts, 
                  :hyperplane_config,
                  :regression_sparsity, :regression_weighted_betas,
                  :regression_lambda, :criterion]
    return merge_kwargs(valid_keys; kwargs...)
end

""" 
    learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::IAI.OptimalTreeLearner, 
                          idxs::Union{Nothing, Array}=nothing; kwargs...)

Wrapper around IAI.fit! for constraint learning.
Arguments:
-lnr: OptimalTreeClassifier or OptimalTreeRegressor
-X: matrix of feature data
-Y: matrix of constraint data.
Returns fitted Learner corresponding to the data. 
NOTE: kwargs from `learn_constraint!` get unpacked here.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::Union{IAI.OptimalTreeLearner, 
                                                                     IAI.Heuristics.RandomForestLearner}, 
                          idxs::Union{Nothing, Array}=nothing; kwargs...)
    n_samples, n_features = size(X);
    @assert n_samples == length(Y);
    # Making sure that we only consider relevant features.
    if !isnothing(idxs)
        IAI.set_params!(lnr, split_features = idxs)
        if typeof(lnr) in [IAI.OptimalTreeRegressor, IAI.Heuristics.RandomForestRegressor]
            IAI.set_params!(lnr, regression_features = idxs)
        end
    else
        IAI.set_params!(lnr, split_features = All())
        if typeof(lnr) in [IAI.OptimalTreeRegressor, IAI.Heuristics.RandomForestRegressor]
            IAI.set_params!(lnr, regression_features = All())
        end
    end
    IAI.fit!(lnr, X, Y; kwargs...)
    return lnr
end

""" Checks that a BlackBoxClassifier has enough feasible/infeasible samples. """
function check_feasibility(bbc::BlackBoxClassifier)
    return bbc.feas_ratio >= get_param(bbc, :threshold_feasibility)
end

check_feasibility(bbr::BlackBoxRegressor) = true

function check_feasibility(gm::GlobalModel)
    return [check_feasibility(bbl) for bbl in gm.bbls]
end

""" Checks that a BlackBoxLearner.learner has adequate accuracy."""
function check_accuracy(bbc::BlackBoxClassifier)
    return bbc.accuracies[end] >= get_param(bbc, :threshold_accuracy)
end

check_accuracy(bbr::BlackBoxRegressor) = 1. # TODO: use MSE?

function check_sampled(bbl::BlackBoxLearner)
    size(bbl.X, 1) == 0 && throw(OCTHaGOnException(string("BlackBoxLearner ", bbl.name, " must be sampled first.")))
    return 
end

""" 
    get_random_trees(lnr::IAI.Heuristics.RandomForestLearner)
Returns one of the trees of RandomForestLearner. 
"""
function get_random_trees(lnr::IAI.Heuristics.RandomForestLearner)
    trees = []
    for i = 1:lnr.num_trees
        tree = IAI.clone(lnr.inner)
        tree.prb_ = lnr.prb_
        tree.tree_ = lnr.forest_.trees[i]
        push!(trees, tree)
    end
    return trees
end

""" 
    boundify(lnr::OptimalTreeRegressor, X::DataFrame, Y, solver::MOI.OptimizerWithAttributes)
    boundify(lnr::IAI.Heuristics.RandomForestRegressor, X::DataFrame, Y, solver::MOI.OptimizerWithAttributes)
    boundify(lnr::OptimalTreeClassifier, X::DataFrame, Y, hypertype::String = "lower", solver::MOI.OptimizerWithAttributes)
Returns bounding hyperplanes of an OptimalTreeLearner.
"""
function boundify(lnr::IAI.OptimalTreeRegressor, X::DataFrame, Y, solver::MOI.OptimizerWithAttributes)
    feas_leaves = find_leaves(lnr)
    leaf_idx = IAI.apply(lnr, X)
    ul_data = Dict()
    for leaf in feas_leaves
        idx = findall(x -> x == leaf, leaf_idx)
        ul_data[-leaf] = u_regress(X[idx,:], Y[idx], solver) # negative leaf shows upper bounding
        ul_data[leaf] = l_regress(X[idx,:], Y[idx], solver)
    end
    return ul_data
end

function boundify(lnr::IAI.OptimalTreeClassifier, X::DataFrame, Y, solver::MOI.OptimizerWithAttributes, hypertype::String = "lower")
    all_leaves = find_leaves(lnr)
    leaf_idx = IAI.apply(lnr, X)
    data = Dict()
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    for leaf in feas_leaves
        idx = findall(x -> x == leaf, leaf_idx)
        if hypertype == "upper"
            data[-leaf] = u_regress(X[idx,:], Y[idx], solver)
        elseif hypertype == "lower"
            data[leaf] = l_regress(X[idx,:], Y[idx], solver)
        end
    end
    return data
end

function boundify(lnr::IAI.Heuristics.RandomForestRegressor,
                  X::DataFrame, Y, solver::MOI.OptimizerWithAttributes)
    trees = get_random_trees(lnr)
    data = Dict()
    for i=1:length(trees)
        ul_data = Dict()
        tree = trees[i]
        all_leaves = find_leaves(tree)
        leaf_idx = IAI.apply(tree, X)
        for leaf in all_leaves
            idx = findall(x -> x == leaf, leaf_idx)
            ul_data[-leaf] = u_regress(X[idx,:], Y[idx], solver) # negative leaf shows upper bounding
            ul_data[leaf] = l_regress(X[idx,:], Y[idx], solver)
        end
        data[i] = ul_data
    end
    return data
end

function boundify(lnr::IAI.Heuristics.RandomForestClassifier,
                    X::DataFrame, Y, solver::MOI.OptimizerWithAttributes, hypertype::String = "lower")
    trees = get_random_trees(lnr)
    data = Dict()
    for i=1:length(trees)
        ul_data = Dict()
        tree = trees[i]
        all_leaves = find_leaves(tree)
        leaf_idx = IAI.apply(tree, X)
        feas_leaves = [j for j in all_leaves if Bool(IAI.get_classification_label(tree, j))]
        for leaf in feas_leaves
            idx = findall(x -> x == leaf, leaf_idx)
            if hypertype == "upper"
                ul_data[-leaf] = u_regress(X[idx,:], Y[idx], solver)
            elseif hypertype == "lower"
                ul_data[leaf] = l_regress(X[idx,:], Y[idx], solver)
            end
        end 
        data[i] = ul_data
    end
    return data
end

"""
    learn_constraint!(bbl::Union{GlobalModel, BlackBoxLearner, Array}; kwargs...)

Constructs a constraint tree from a BlackBoxLearner and dumps in bbo.learners.
Arguments:
    bbl: OCT structs (GM, bbl, Array{bbl})
    kwargs: arguments for learners and fits.
"""
function learn_constraint!(bbc::BlackBoxClassifier; kwargs...)
    check_sampled(bbc)
    set_param(bbc, :reloaded, false) # Makes sure that we know trees are retrained. 
    lnr = base_classifier()
    IAI.set_params!(lnr; minbucket = 
        maximum([2*length(bbc.vars)/length(bbc.Y), lnr.minbucket]))
    if bbc.equality # Equalities are approximated more aggressively.
        IAI.set_params!(lnr; max_depth = 6, ls_num_tree_restarts = 30, fast_num_support_restarts = 15)
    end
    IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
    if check_feasibility(bbc) || get_param(bbc, :ignore_feasibility)
        lnr = learn_from_data!(bbc.X, bbc.Y .>= 0, lnr; fit_classifier_kwargs(; kwargs...)...)
        push!(bbc.learners, lnr)
        push!(bbc.accuracies, IAI.score(lnr, bbc.X, bbc.Y .>= 0)) # TODO: add ability to specify criterion. 
        push!(bbc.learner_kwargs, Dict(kwargs))
    else
        throw(OCTHaGOnException("Not enough feasible samples for BlackBoxClassifier " * string(bbc.name) * "."))
    end
    return
end

function learn_constraint!(bbr::BlackBoxRegressor, threshold::Pair = Pair("reg", nothing);  kwargs...)
    classifications = ["upper", "lower", "upperlower"]
    check_sampled(bbr)
    get_param(bbr, :reloaded) && set_param(bbr, :reloaded, false) # Makes sure that we know trees are retrained. 
    if bbr.convex && !bbr.equality
        return # If convex, don't train a tree!
    elseif threshold.first in classifications
        lnr = base_classifier()
        IAI.set_params!(lnr; minbucket = 
            maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]), 
            classifier_kwargs(; kwargs...)...)
        ul_data = Dict()
        if threshold.first == "upper" # Upper bounding classifier with upper bounds in leaves
            lnr = learn_from_data!(bbr.X, bbr.Y .<= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
            merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT, "upper"))
        elseif threshold.first == "lower" # Lower bounding classifier with lower bounds in leaves
            lnr = learn_from_data!(bbr.X, bbr.Y .>= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
            merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT, "lower"))
        elseif threshold.first == "upperlower" # Upper bounding classifier with bounds in each leaf
            lnr = learn_from_data!(bbr.X, bbr.Y .<= threshold.second, lnr; fit_classifier_kwargs(; kwargs...)...)
            merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT, "lower"))
            merge!(ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT, "upper"))
        end
        push!(bbr.learners, lnr);
        push!(bbr.learner_kwargs, Dict(kwargs))
        push!(bbr.thresholds, threshold)
        push!(bbr.ul_data, ul_data)
        push!(bbr.accuracies, IAI.score(lnr, bbr.X, bbr.Y .>= 0))
    elseif threshold.first == "reg"
        lnr = base_regressor()
        IAI.set_params!(lnr; minbucket = 
            maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]),
            regressor_kwargs(; kwargs...)...)
        if bbr.equality # Equalities cannot leverage convexity unfortunately...
            if threshold.second == nothing
                lnr = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
            else
                idxs = findall(y -> y <= threshold.second, bbr.Y) 
                lnr = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
            end        
        elseif threshold.second == nothing && bbr.local_convexity < 0.75
            lnr = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
        elseif threshold.second == nothing && bbr.local_convexity >= 0.75
            lnr = learn_from_data!(bbr.X, bbr.curvatures .> 0, lnr; fit_regressor_kwargs(; kwargs...)...)             
        elseif bbr.local_convexity < 0.75
            idxs = findall(y -> y <= threshold.second, bbr.Y) 
            lnr = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
        elseif bbr.local_convexity >= 0.75
            idxs = findall(y -> y <= threshold.second, bbr.Y) 
            lnr = learn_from_data!(bbr.X[idxs, :], bbr.curvatures[idxs] .> 0, lnr; fit_regressor_kwargs(; kwargs...)...)
        end
        push!(bbr.learners, lnr);
        push!(bbr.learner_kwargs, Dict(kwargs))
        push!(bbr.thresholds, threshold)
        push!(bbr.ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT))
        push!(bbr.accuracies, IAI.score(lnr, bbr.X, bbr.Y))
    elseif threshold.first == "rfreg"
        lnr = base_rf_regressor()
        bbr.local_convexity < 0.75 || throw(OCTHaGOnException("Cannot use RandomForestRegressor " *
        "on BBR $(bbr.name) since it is almost convex."))
        IAI.set_params!(lnr; minbucket = 
            maximum([2*length(bbr.vars)/length(bbr.Y), lnr.minbucket]),
            regressor_kwargs(; kwargs...)...)
        if threshold.second == nothing
            lnr = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)   
        else
            idxs = findall(y -> y <= threshold.second, bbr.Y) 
            lnr = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs], lnr; fit_regressor_kwargs(; kwargs...)...)
        end
        push!(bbr.learners, lnr);
        push!(bbr.learner_kwargs, Dict(kwargs))
        push!(bbr.thresholds, threshold)
        push!(bbr.ul_data, boundify(lnr, bbr.X, bbr.Y, SOLVER_SILENT))
        push!(bbr.accuracies, IAI.score(lnr, bbr.X, bbr.Y))
    else
        throw(OCTHaGOnException("$(threshold.first) is not a valid learner type for" *
            " thresholded learning of BBR $(bbr.name)."))
    end    
    return
end

function learn_constraint!(bbl::Array; kwargs...)
   for fn in bbl
        learn_constraint!(fn; kwargs...)
   end
end

learn_constraint!(gm::GlobalModel; kwargs...) = 
    learn_constraint!(gm.bbls; kwargs...)
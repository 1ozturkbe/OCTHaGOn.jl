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

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! """
function fit_classifier_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}(:sample_weight => :autobalance) # default kwargs
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

""" Wrapper around IAI.fit! for constraint learning.
Arguments:
    lnr: OptimalTreeClassifier or OptimalTreeRegressor
    X: matrix of feature data
    Y: matrix of constraint data.
Returns:
    lnr: Fitted Learner corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, lnr::IAI.OptimalTreeLearner, 
                          idxs::Union{Nothing, Array}=nothing; kwargs...)
    n_samples, n_features = size(X);
    @assert n_samples == length(Y);
    # Making sure that we only consider relevant features.
    if !isnothing(idxs)
        IAI.set_params!(lnr, split_features = idxs)
        if typeof(lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(lnr, regression_features = idxs)
        end
    else
        IAI.set_params!(lnr, split_features = All())
        if typeof(lnr) == IAI.OptimalTreeRegressor
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

check_accuracy(bbr::BlackBoxRegressor) = 1. # TODO: use MSE

function check_sampled(bbl::BlackBoxLearner)
    size(bbl.X, 1) == 0 && throw(OCTException(string("BlackBoxLearner ", bbl.name, " must be sampled first.")))
    return 
end

""" 
    ul_boundify(lnr::OptimalTreeLearner, X::DataFrame, Y; solver = CPLEX_SILENT)

Returns upper/lower bound data of an OptimalTreeLearner.
"""
function ul_boundify(lnr::IAI.OptimalTreeLearner, X::DataFrame, Y; solver = CPLEX_SILENT)
    all_leaves = find_leaves(lnr)
    leaf_idx = IAI.apply(lnr, X)
    ul_data = Dict()
    feas_leaves = []
    if lnr isa IAI.OptimalTreeClassifier
        feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    elseif lnr isa IAI.OptimalTreeRegressor
        feas_leaves = all_leaves # TODO: improve depending on thresholding. 
    end
    for leaf in feas_leaves
        idx = findall(x -> x == leaf, leaf_idx)
        (α0, α), (β0, β) = ul_regress(X[idx,:], Y[idx]; solver = solver)
        ul_data[leaf] = [(α0, α), (β0, β)]
    end
    return ul_data
end

"""
    learn_constraint!(bbl::Union{GlobalModel, BlackBoxLearner, Array}; kwargs...)

Constructs a constraint tree from a BlackBoxLearner and dumps in bbo.learners.
Arguments:
    bbl: OCT structs (GM, bbl, Array{bbl})
    ignore_feas::Bool: Whether to ignore feasibility thresholds for BlackBoxClassifiers.
    kwargs: arguments for learners and fits.
"""
function learn_constraint!(bbc::BlackBoxClassifier, ignore_feas::Bool = false; kwargs...)
    check_sampled(bbc)
    set_param(bbc, :reloaded, false) # Makes sure that we know trees are retrained. 
    lnr = base_classifier()
    IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
    if check_feasibility(bbc) || ignore_feas
        nl = learn_from_data!(bbc.X, bbc.Y .>= 0, lnr; fit_classifier_kwargs(; kwargs...)...)
        push!(bbc.learners, nl)
        push!(bbc.accuracies, IAI.score(nl, bbc.X, bbc.Y .>= 0)) # TODO: add ability to specify criterion. 
        push!(bbc.learner_kwargs, Dict(kwargs))
    else
        throw(OCTException("Not enough feasible samples for BlackBoxClassifier " * string(bbc.name) * "."))
    end
    return
end

function learn_constraint!(bbr::BlackBoxRegressor, ignore_feas::Bool = false; kwargs...)
    check_sampled(bbr)
    set_param(bbr, :reloaded, false) # Makes sure that we know trees are retrained. 
    if haskey(kwargs, :threshold)
        lnr = base_classifier()
        IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
        nl = learn_from_data!(bbr.X, bbr.Y .<= kwargs[:threshold], lnr; fit_classifier_kwargs(; kwargs...)...)    
        push!(bbr.learners, nl);
        push!(bbr.learner_kwargs, Dict(kwargs))
        push!(bbr.thresholds, kwarg[:threshold])
        push!(bbr.ul_data, ul_boundify(nl, bbr.X, bbr.Y))
        return 
    end
    lnr = base_regressor()
    IAI.set_params!(lnr; regressor_kwargs(; kwargs...)...)
    nl = learn_from_data!(bbr.X, bbr.Y, lnr; fit_regressor_kwargs(; kwargs...)...)             
    push!(bbr.learners, nl);
    push!(bbr.learner_kwargs, Dict(kwargs))
    push!(bbr.thresholds, nothing)  
    if haskey(kwargs, :regression_sparsity) && kwargs[:regression_sparsity] != :all
        push!(bbr.ul_data, ul_boundify(nl, bbr.X, bbr.Y))
    else
        push!(bbr.ul_data, Dict())
    end
    return
end

function learn_constraint!(bbl::Array, ignore_feas::Bool = false; kwargs...)
   for fn in bbl
        learn_constraint!(fn, ignore_feas; kwargs...)
   end
end

learn_constraint!(gm::GlobalModel, ignore_feas::Bool = get_param(gm, :ignore_feasibility); kwargs...) = 
    learn_constraint!(gm.bbls, ignore_feas; kwargs...)

"""
    save_fit(bbl::Union{GlobalModel, BlackBoxLearner, Array}, dir::String)

Saves IAI fits associated with different OptimalConstraintTree objects.
"""
function save_fit(bbl::BlackBoxLearner, dir::String = TREE_DIR)
    IAI.write_json(dir * bbl.name * ".json", bbl.learners[end])
end

save_fit(bbls::Array, dir::String = TREE_DIR) = [save_fit(bbl, dir) for bbl in bbls]
save_fit(gm::GlobalModel, dir::String = TREE_DIR) = save_fit(gm.bbls, dir * "$(gm.name)_")

"""
    load_fit(BlackBoxLearner, dir::String = TREE_DIR)

Loads IAI fits associated with OptimalConstraintTree objects.
Checks that there is correspondence between loaded trees and the associated constraints.
"""

function load_fit(bbl::BlackBoxLearner, dir::String = TREE_DIR)
    loaded_lnr = IAI.read_json(dir * bbl.name * ".json");
    size(IAI.variable_importance(loaded_lnr), 1) == length(bbl.vars) || throw(
        OCTException("Object " * bbl.name * " does not match associated learner."))
    set_param(bbl, :reloaded, true)
    push!(bbl.learners, loaded_lnr)
end

load_fit(bbls::Array{BlackBoxLearner}, 
        dir::String = TREE_DIR) = [load_fit(bbl, dir) for bbl in bbls]

load_fit(gm::GlobalModel, dir::String = TREE_DIR) = load_fit(gm.bbls, dir * "$(gm.name)_")
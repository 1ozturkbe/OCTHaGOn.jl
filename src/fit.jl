""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! """
function fit_regressor_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}(:validation_criterion => :mse)
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Preprocesses merging of kwargs for IAI.fit!, to avoid errors! """
function fit_classifier_kwargs(; kwargs...)
    nkwargs = Dict{Symbol, Any}(:validation_criterion => :misclassification,
                    :sample_weight => :autobalance) # default kwargs
    for item in kwargs
        if item.first in keys(nkwargs)
            nkwargs[item.first] = item.second
        end
    end
    if nkwargs[:validation_criterion] == :sensitivity
        delete!(nkwargs, :sample_weight)
        nkwargs[:positive_label] = 1
    end
    return nkwargs
end

""" Function that preprocesses merging of kwargs for IAI.OptimalTreeLearner!, 
    to avoid errors! """
function lnr_kwargs(regress::Bool = false; kwargs...)
    nkwargs = Dict{Symbol, Any}() 
    valid_keys = [:random_seed, :max_depth, :cp, :minbucket, 
                  :fast_num_support_restarts, :localsearch, 
                  :ls_num_hyper_restarts, :ls_num_tree_restarts, 
                  :hyperplane_config]
    if regress
        append!(valid_keys, [:regression_sparsity, :regression_weighted_betas,
                             :regression_lambda])   
    end    
    for item in kwargs
        if item.first in valid_keys
            nkwargs[item.first] = item.second
        end
    end
    return nkwargs
end

""" Wrapper around IAI.GridSearch for constraint learning.
Arguments:
    lnr: Unfit OptimalTreeClassifier, OptimalTreeRegressor or Grid
    X: matrix of feature data
    Y: matrix of constraint data.
Returns:
    lnr: list of Fitted Grids corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, grid::Union{IAI.OptimalTreeLearner, IAI.GridSearch}, 
                          idxs::Union{Nothing, Array}=nothing; kwargs...)
    # TODO: fix grid vs. solo learner issues. 
    n_samples, n_features = size(X);
    @assert n_samples == length(Y);
    # Making sure that we only consider relevant features.
    if !isnothing(idxs)
        IAI.set_params!(grid.lnr, split_features = idxs)
        if typeof(grid.lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(grid.lnr, regression_features = idxs)
        end
    else
        IAI.set_params!(grid.lnr, split_features = All())
        if typeof(grid.lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(grid.lnr, regression_features = All())
        end
    end
    IAI.fit!(grid, X, Y; kwargs...)
    return grid
end

""" Checks that a BlackBoxClassifier has enough feasible/infeasible samples. """
function check_feasibility(bbc::BlackBoxClassifier)
    return bbc.feas_ratio >= get_param(bbc, :threshold_feasibility)
end

check_feasibility(bbr::BlackBoxRegressor) = true

function check_feasibility(gm::GlobalModel)
    return [check_feasibility(bbf) for bbf in gm.bbfs]
end

""" Checks that a BlackBoxLearner.learner has adequate accuracy."""
function check_accuracy(bbc::BlackBoxClassifier)
    return bbc.accuracies[end] >= get_param(bbc, :threshold_accuracy)
end

check_accuracy(bbr::BlackBoxRegressor) = true # TODO: use MSE

function check_sampled(bbf::Union{BlackBoxClassifier, BlackBoxRegressor})
    size(bbf.X, 1) == 0 && throw(OCTException(string("BlackBoxLearner ", bbf.name, " must be sampled first.")))
end

"""
    learn_constraint!(bbf::Union{GlobalModel, BlackBoxLearner, Array}; kwargs...)

Constructs a constraint tree from a BlackBoxLearner and dumps in bbo.learners.
Arguments:
    bbf: OCT structs (GM, BBF, Array{BBF})
    ignore_feas::Bool: Whether to ignore feasibility thresholds for BlackBoxClassifiers.
    kwargs: arguments for learners and fits.
"""
function learn_constraint!(bbc::BlackBoxClassifier, ignore_feas::Bool = false; kwargs...)
    check_sampled(bbc)
    set_param(bbc, :reloaded, false) # Makes sure that we know trees are retrained. 
    lnr = base_classifier()
    IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
    if check_feasibility(bbc) || ignore_feas
        nl = learn_from_data!(bbc.X, bbc.Y .>= 0,
                            gridify(lnr);
                            fit_classifier_kwargs(; kwargs...)...)
        push!(bbc.learners, nl)
        bbc.predictions = IAI.predict(nl, bbc.X)
        push!(bbc.accuracies, IAI.score(nl, bbc.X, bbc.Y .>= 0, criterion = nl.criterion))
        push!(bbc.learner_kwargs, Dict(kwargs))
    else
        throw(OCTException("Not enough feasible samples for BlackBoxClassifier " * string(bbc.name) * "."))
    end
    return
end

function learn_constraint!(bbr::BlackBoxRegressor, ignore_feas::Bool = false; kwargs...)
    check_sampled(bbr)
    set_param(bbr, :reloaded, false) # Makes sure that we know trees are retrained. 
    lnr = base_regressor()
    IAI.set_params!(lnr; regressor_kwargs(; kwargs...)...)
    nl = learn_from_data!(bbr.X[idxs, :], bbr.Y[idxs],
                            gridify(lnr);
                            fit_regressor_kwargs(; kwargs...)...)             
    push!(bbr.learners, nl);
    bbc.predictions = IAI.predict(nl, bbc.X)
    push!(bbr.accuracies, IAI.score(nl, bbr.X, bbr.Y, criterion = nl.criterion))
    push!(bbr.learner_kwargs, Dict(kwargs))
    return
end

function learn_constraint!(bbf::Array, ignore_feas::Bool = false; kwargs...)
   for fn in bbf
        learn_constraint!(fn, ignore_feas; kwargs...)
   end
end

learn_constraint!(gm::GlobalModel, ignore_feas::Bool = get_param(gm, :ignore_feasibility); kwargs...) = 
    learn_constraint!(gm.bbfs, ignore_feas; kwargs...)

"""
    save_fit(bbf::Union{GlobalModel, BlackBoxClassifier, BlackBoxRegressor, Array}, dir::String)

Saves IAI fits associated with different OptimalConstraintTree objects.
"""
function save_fit(bbf::Union{BlackBoxClassifier, BlackBoxRegressor}, dir::String = SAVE_DIR)
    IAI.write_json(dir * bbf.name * ".json", bbf.learners[end])
end

save_fit(bbfs::Array, dir::String = SAVE_DIR) = [save_fit(bbf, dir) for bbf in bbfs]
save_fit(gm::GlobalModel, dir::String = SAVE_DIR) = save_fit(gm.bbfs, dir)

"""
    load_fit(BlackBoxLearner, dir::String = SAVE_DIR)

Loads IAI fits associated with OptimalConstraintTree objects.
Checks that there is correspondence between loaded trees and the associated constraints.
"""

function load_fit(bbf::Union{BlackBoxClassifier, BlackBoxRegressor}, dir::String = SAVE_DIR)
    loaded_grid = IAI.read_json(dir * bbf.name * ".json");
    size(IAI.variable_importance(loaded_grid.lnr), 1) == length(bbf.vars) || throw(
        OCTException("Object " * bbf.name * " does not match associated learner."))
    set_param(bbf, :reloaded, true)
    push!(bbf.learners, loaded_grid)
end

load_fit(bbfs::Array{Union{BlackBoxClassifier, BlackBoxRegressor}}, 
        dir::String = SAVE_DIR) = [load_fit(bbf, dir) for bbf in bbfs]

load_fit(gm::GlobalModel, dir::String = SAVE_DIR) = load_fit(gm.bbfs, dir)
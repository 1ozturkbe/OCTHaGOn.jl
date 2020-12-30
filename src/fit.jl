""" Default fitting kwargs for OCTs. """
function default_fit_kwargs()
    Dict(:validation_criterion => :misclassification,
         :sample_weight => :autobalance)
end

""" Function that preprocesses merging of fit kwargs, 
    to avoid errors! """
function merge_fit_kwargs(kwargs1, kwargs2)
    nkwargs = merge(kwargs1, kwargs2)
    if haskey(nkwargs, :validation_criterion) && nkwargs[:validation_criterion] == :sensitivity
        if haskey(nkwargs, :sample_weight)
            delete!(nkwargs, :sample_weight)
        end
        if !haskey(kwargs, :positive_label)
            nkwargs[:positive_label] = 1
        end
    end
    return nkwargs
end

""" Wrapper around IAI.GridSearch for constraint learning.
Arguments:
    lnr: Unfit OptimalTreeClassifier or Grid
    X: matrix of feature data
    Y: matrix of constraint data.
Returns:
    lnr: list of Fitted Grids corresponding to the data
NOTE: kwargs get unpacked here, all the way from learn_constraint!.
"""
function learn_from_data!(X::DataFrame, Y::AbstractArray, grid, idxs::Union{Nothing, Array}=nothing; kwargs...)
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

""" Checks that a BlackBoxFunction has enough feasible/infeasible samples. """
function check_feasibility(bbf::Union{BlackBoxFunction, DataConstraint})
    return bbf.feas_ratio >= get_param(bbf, :threshold_feasibility)
end

function check_feasibility(gm::GlobalModel)
    return [check_feasibility(bbf) for bbf in gm.bbfs]
end

""" Checks that a BlackBoxFunction.learner has adequate accuracy."""
function check_accuracy(bbf::Union{BlackBoxFunction, DataConstraint})
    return bbf.accuracies[end] >= get_param(bbf, :threshold_accuracy)
end

function check_accuracy(gm::GlobalModel)
    return [check_accuracy(bbf) for bbf in gm.bbfs]
end

"""
    learn_constraint!(bbf::Union{GlobalModel, Array{BlackBoxFunction}, BlackBoxFunction},
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           ignore_checks::Bool = false or gm.settings[:ignore_feasibility]; kwargs...)

Constructs a constraint tree from a BlackBoxFunction and dumps in bbf.learners.
Arguments:
    bbf:: OCT structs (GM, BBF, Array{BBF})
    lnr: Unfit OptimalTreeClassifier or Grid
    X: new data to add to BlackBoxFunction and evaluate
    kwargs: arguments for learn_from_data!
Returns:
    nothing
"""
function learn_constraint!(bbf::Union{BlackBoxFunction, DataConstraint},
                           lnr::IAI.OptimalTreeLearner = base_otc(localsearch = get_param(bbf, :localsearch)),
                           ignore_checks::Bool = false; kwargs...)
    if isa(bbf.X, Nothing)
        throw(OCTException(string("BlackBoxFn ", bbf.name, " must be sampled first.")))
    end
    n_samples, n_features = size(bbf.X)
    if bbf.feas_ratio == 1.0
        return
    elseif check_feasibility(bbf) || ignore_checks
        nkwargs = merge_fit_kwargs(default_fit_kwargs(), kwargs)
        nl = learn_from_data!(bbf.X, bbf.Y .>= 0,
                              gridify(lnr);
                              nkwargs...)
        push!(bbf.learners, nl);
        push!(bbf.accuracies, IAI.score(nl, bbf.X, bbf.Y .>= 0))
        push!(bbf.learner_kwargs, nkwargs)
    else
        @warn("Not enough feasible samples for constraint " * string(bbf.name) * ".")
    end
    return
end

function learn_constraint!(bbf::Array,
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           ignore_checks::Bool = false; kwargs...)
   for fn in bbf
        learn_constraint!(fn, lnr, ignore_checks; kwargs...)
   end
end

function learn_constraint!(gm::GlobalModel,
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           ignore_checks::Bool = get_param(gm, :ignore_feasibility); kwargs...)
   set_param(gm, :ignore_feasibility, ignore_checks) # update check settings
   learn_constraint!(gm.bbfs, lnr, ignore_checks; kwargs...)
end

"""
Basic regression purely for debugging.
TODO: refine and/or remove.
"""
function regress(points::DataFrame, values::Array; weights::Array = ones(length(values)))
    lnr= IAI.OptimalFeatureSelectionRegressor(sparsity = :all); # TODO: optimize regression method.
    IAI.fit!(lnr, points, values, sample_weight=weights)
    return lnr
end


"""
    save_fit(bbf::Union{GlobalModel, BlackBoxFunction, DataConstraint, Array}, dir::String)

Saves IAI fits associated with different OptimalConstraintTree objects.
"""
function save_fit(bbf::Union{BlackBoxFunction, DataConstraint}, dir::String = SAVE_DIR)
    IAI.write_json(dir * bbf.name * ".json", bbf.learners[end])
end

save_fit(bbfs::Array, dir::String = SAVE_DIR) = [save_fit(bbf, dir) for bbf in bbfs]
save_fit(gm::GlobalModel, dir::String = SAVE_DIR) = save_fit(gm.bbfs, dir)

"""
    load_fit(bbf::Union{BlackBoxFunction}, DataConstraint, dir::String = SAVE_DIR)

Loads IAI fits associated with OptimalConstraintTree objects.
Checks that there is correspondence between loaded trees and the associated constraints.
"""
function load_fit(bbf::Union{BlackBoxFunction, DataConstraint}, dir::String = SAVE_DIR)
    loaded_grid = IAI.read_json(dir * bbf.name * ".json");
    size(IAI.variable_importance(loaded_grid.lnr), 1) == length(bbf.vars) || throw(
        OCTException("Object " * bbf.name * " does not match associated learner."))
    set_param(bbf, :reloaded, true)
    push!(bbf.learners, loaded_grid)
end

load_fit(bbfs::Array, dir::String = SAVE_DIR) = [load_fit(bbf, dir) for bbf in bbfs]
load_fit(gm::GlobalModel, dir::String = SAVE_DIR) = load_fit(gm.bbfs, dir)
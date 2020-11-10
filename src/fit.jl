function learn_from_data!(X::DataFrame, Y::AbstractArray, grid; idxs::Union{Nothing, Array}=nothing,
                         weights = :autobalance,
                         validation_criterion=:misclassification)
    """ Wrapper around IAI.GridSearch for constraint learning.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        X: matrix of feature data
        Y: matrix of constraint data.
    Returns:
        lnr: list of Fitted Grids corresponding to the data
    NOTE: All constraints must take in full vector of X values.
    """
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
    IAI.fit!(grid, X, Y,
             validation_criterion = :misclassification, sample_weight=weights);
    return grid
end

""" Checks that a BlackBoxFunction has enough feasible/infeasible samples. """
function check_feasibility(bbf::Union{GlobalModel, BlackBoxFunction})
    if isa(bbf, BlackBoxFunction)
        return bbf.feas_ratio >= bbf.threshold_feasibility
    else
        return [check_feasibility(fn) for fn in bbf.bbfs]
    end
end

""" Checks that a BlackBoxFunction.learner has adequate accuracy."""
function check_accuracy(bbf::Union{GlobalModel, BlackBoxFunction})
    if isa(bbf, BlackBoxFunction)
        return bbf.accuracies[end] >= bbf.threshold_accuracy
    else
        return [check_accuracy(fn) for fn in bbf.bbfs]
    end
end

""" Classifies and returns names of functions that pass/fail the feasibility check. """
function fns_by_feasibility(gm::GlobalModel)
    arr = [check_feasibility(fn) for fn in gm.bbfs]
    infeas_idxs = findall(x -> x .== 0, arr)
    feas_idxs = findall(x -> x .!= 0, arr) # TODO: use complement.
    return gm.bbfs[feas_idxs], gm.bbfs[infeas_idxs]
end

"""
    learn_constraint!(bbf::Union{GlobalModel, Array{BlackBoxFunction}, BlackBoxFunction};
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false)

Constructs a constraint tree from a BlackBoxFunction and dumps in bbf.learners.
Arguments:
    bbf:: OCT structs (GM, BBF, Array{BBF})
    lnr: Unfit OptimalTreeClassifier or Grid
    X: new data to add to BlackBoxFunction and evaluate
    weights: weighting of the data points
    dir: save location
    ignore_checks: True only for debugging purposes.
                   Ignores feasibility and sample distribution checks.
Returns:
    nothing
"""
function learn_constraint!(bbf::Union{GlobalModel, Array{BlackBoxFunction}, BlackBoxFunction};
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false)
    if isa(bbf, GlobalModel)
        for fn in bbf.bbfs
            learn_constraint!(fn, lnr=lnr, weights=weights, dir=dir,
                              validation_criterion = validation_criterion, ignore_checks=ignore_checks)
        end
        return
    elseif isa(bbf, Array{BlackBoxFunction})
        for fn in bbf
            learn_constraint!(fn, lnr=lnr, weights=weights, dir=dir,
                              validation_criterion = validation_criterion, ignore_checks=ignore_checks)
        end
        return
    end
    if isa(bbf.X, Nothing)
        throw(OCTException(string("BlackBoxFn ", bbf.name, " must be sampled first.")))
    end
    n_samples, n_features = size(bbf.X)
    if bbf.feas_ratio == 1.0
        return
    elseif check_feasibility(bbf) || ignore_checks
        # TODO: optimize Matrix/DataFrame conversion. Perhaps change the choice.
        nl = learn_from_data!(bbf.X, bbf.Y .>= 0,
                              gridify(lnr),
                              weights=weights,
                              validation_criterion=:misclassification);
        push!(bbf.learners, nl);
        push!(bbf.accuracies, IAI.score(nl, Matrix(bbf.X), bbf.Y .>= 0))
    else
        @warn("Not enough feasible samples for constraint " * string(bbf.name) * ".")
    end
    if dir != "-"
        IAI.write_json(string(dir, bbf.name, "_tree_", length(bbf.learners), ".json"),
                           bbf.learners[end]);
    end
    return
end

"""
Basic regression purely for debugging.
TODO: refine and/or remove.
"""
function regress(points::DataFrame, values::Array; weights::Union{Array, Nothing} = nothing)
    lnr= IAI.OptimalFeatureSelectionRegressor(sparsity = :all); # TODO: optimize regression method.
    if isnothing(weights)
        weights = ones(length(values));
    end
    IAI.fit!(lnr, points, values, sample_weight=weights)
    return lnr
end

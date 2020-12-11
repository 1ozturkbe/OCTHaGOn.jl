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
function check_feasibility(bbf::Union{BlackBoxFunction, DataConstraint})
    return bbf.feas_ratio >= bbf.threshold_feasibility
end

function check_feasibility(gm::GlobalModel)
    return [check_feasibility(bbf) for bbf in gm.bbfs]
end

""" Checks that a BlackBoxFunction.learner has adequate accuracy."""
function check_accuracy(bbf::Union{BlackBoxFunction, DataConstraint})
    return bbf.accuracies[end] >= bbf.threshold_accuracy
end

function check_accuracy(gm::GlobalModel)
    return [check_accuracy(bbf) for bbf in gm.bbfs]
end

"""
    learn_constraint!(bbf::Union{GlobalModel, Array{BlackBoxFunction}, BlackBoxFunction};
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false or gm.settings[:ignore_feasibility])

Constructs a constraint tree from a BlackBoxFunction and dumps in bbf.learners.
Arguments:
    bbf:: OCT structs (GM, BBF, Array{BBF})
    lnr: Unfit OptimalTreeClassifier or Grid
    X: new data to add to BlackBoxFunction and evaluate
    weights: weighting of the data points
    dir: save location
Returns:
    nothing
"""
function learn_constraint!(bbf::Union{BlackBoxFunction, DataConstraint};
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false)
    if isa(bbf.X, Nothing)
        throw(OCTException(string("BlackBoxFn ", bbf.name, " must be sampled first.")))
    end
    n_samples, n_features = size(bbf.X)
    if bbf.feas_ratio == 1.0
        return
    elseif check_feasibility(bbf) || ignore_checks
        nl = learn_from_data!(bbf.X, bbf.Y .>= 0,
                              gridify(lnr),
                              weights=weights,
                              validation_criterion=:misclassification);
        push!(bbf.learners, nl);
        push!(bbf.accuracies, IAI.score(nl, bbf.X, bbf.Y .>= 0))
    else
        @warn("Not enough feasible samples for constraint " * string(bbf.name) * ".")
    end
    if dir != "-"
        IAI.write_json(string(dir, bbf.name, "_tree_", length(bbf.learners), ".json"),
                           bbf.learners[end]);
    end
    return
end

function learn_constraint!(bbf::Array;
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false)
   for fn in bbf
        learn_constraint!(fn, lnr=lnr, weights=weights, dir=dir,
                          validation_criterion = validation_criterion, ignore_checks = ignore_checks)
   end
end

function learn_constraint!(gm::GlobalModel;
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = gm.settings[:ignore_feasibility])
   gm.settings[:ignore_feasibility] = ignore_checks # update check settings
   learn_constraint!(gm.bbfs, lnr=lnr, weights=weights, dir=dir,
                          validation_criterion = validation_criterion, ignore_checks = ignore_checks)
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

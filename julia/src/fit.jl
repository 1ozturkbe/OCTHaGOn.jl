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
        IAI.set_params!(grid.lnr, split_features = :all)
        if typeof(grid.lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(grid.lnr, regression_features=:all)
        end
    end
    IAI.fit!(grid, X, Y,
             validation_criterion = :misclassification, sample_weight=weights);
    return grid
end

function feasibility_check(bbf::Union{ModelData, BlackBoxFunction})
    """ Checks that a BlackBoxFunction has enough feasible/infeasible samples. """
    if isa(bbf, BlackBoxFunction)
        return bbf.feas_ratio >= bbf.threshold_feasibility
    else
        return [feasibility_check(fn) for fn in bbf.fns]
    end
end

function accuracy_check(bbf::Union{ModelData, BlackBoxFunction})
    """ Checks that a BlackBoxFunction.learner has adequate accuracy."""
    if isa(bbf, BlackBoxFunction)
        return bbf.accuracies[end] >= bbf.threshold_accuracy
    else
        return [accuracy_check(fn) for fn in bbf.fns]
    end
end

function fns_by_feasibility(md::ModelData)
    """Classifies and returns names of functions that pass/fail the feasibility check. """
    arr = [feasibility_check(fn) for fn in md.fns];
    infeas_idxs = findall(x -> x .== 0, arr);
    feas_idxs = findall(x -> x .!= 0, arr);
    names = [fn.name for fn in md.fns];
    return names[feas_idxs], names[infeas_idxs]
end

function learn_constraint!(bbf::Union{ModelData, BlackBoxFunction};
                           lnr::IAI.OptimalTreeLearner = base_otc(),
                           weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                           validation_criterion=:misclassification,
                           ignore_checks::Bool = false)
    """
    Return a constraint tree from a BlackBoxFunction.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        constraint: BlackBoxFunction in std form (>= 0)
        X: new data to add to BlackBoxFunction and evaluate
    Returns:
        lnr: Fitted Grid
    """
    if isa(bbf, ModelData)
        for fn in bbf.fns
            learn_constraint!(fn, lnr=lnr, weights=weights, dir=dir,
                              validation_criterion = validation_criterion, ignore_checks=ignore_checks)
        end
        return
    end
    if isa(bbf.X, Nothing)
        sample_and_eval!(bbf)
    end
    n_samples, n_features = size(bbf.X)
    if feasibility_check(bbf) || ignore_checks
        # TODO: optimize Matrix/DataFrame conversion. Perhaps change the choice.
        nl = learn_from_data!(bbf.X, bbf.Y .>= 0,
                              gridify(lnr),
                              weights=weights,
                              validation_criterion=:misclassification);
        push!(bbf.learners, nl);
        push!(bbf.accuracies, IAI.score(nl, Matrix(bbf.X), bbf.Y .>= 0))
    else
        @warn "Not enough feasible samples."
    end
    if dir != "-"
        IAI.write_json(string(dir, bbf.name, "_tree_", length(bbf.learners), ".json"),
                           bbf.learners[end]);
    end
end

function regress(points::DataFrame, values::Array; weights::Union{Array, Nothing} = nothing)
    lnr= IAI.OptimalFeatureSelectionRegressor(sparsity = :all); # TODO: optimize regression method.
    if isnothing(weights)
        weights = ones(length(values));
    end
    IAI.fit!(lnr, points, values, sample_weight=weights)
    return lnr
end

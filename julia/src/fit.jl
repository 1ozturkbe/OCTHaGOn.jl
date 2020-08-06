using Gurobi
using JuMP

include("constraintify.jl")
include("exceptions.jl")
include("model_data.jl")
include("black_box_function.jl")
include("learners.jl")

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

function feasibility_check(bbf::BlackBoxFn)
    return bbf.feas_ratio >= bbf.threshold_feasibility && bbf.feas_ratio <= 1 - bbf.threshold_feasibility
end

function accuracy_check(bbf::BlackBoxFn)
    return bbf.accuracies[end] >= bbf.threshold_accuracy
end

function learn_constraint!(bbf::BlackBoxFn; lnr::IAI.OptimalTreeLearner = base_otc(),
                                                weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                                                validation_criterion=:misclassification)
    """
    Return a constraint tree from a BlackBoxFn.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        constraint: BlackBoxFn in std form (>= 0)
        X: new data to add to BlackBoxFn and evaluate
    Returns:
        lnr: Fitted Grid
    """
    if isa(bbf.X, Nothing)
        sample_and_eval!(bbf)
    end
    n_samples, n_features = size(bbf.X)
    if feasibility_check(bbf)
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

function fit!(md::ModelData; X::Union{Array, Nothing} = nothing,
                            n_samples = 1000, jump_model::Union{JuMP.Model, Nothing} = nothing,
                            lnr::IAI.Learner=base_otc(),
                            weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                            validation_criterion::Symbol = :misclassification,
                            return_samples = false)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    if isnothing(X)
        X = sample(md, n_samples=n_samples);
    else
       n_samples = size(X, 1);
    end
    n_features = length(md.c);
    trees, X = learn_constraints!(lnr, md.fns, X, weights = weights,
                                    validation_criterion=validation_criterion,
                                    return_samples=true);

    if dir != "-"
        for i=1:size(trees, 1)
            IAI.write_json(string(dir, "_tree_", i, ".json"),
                           trees[i]);
        end
    end
    return trees
end

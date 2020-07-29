using Gurobi
using JuMP

include("constraintify.jl")
include("exceptions.jl")
include("model_data.jl")
include("black_box_function.jl")
include("learners.jl")

function learn_from_data!(X::AbstractArray, Y::AbstractArray, grid; idxs=Union{Nothing, Array},
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
            IAI.set_params!(grid.lnr, regression_features=idxs)
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

function learn_constraints!(lnr::IAI.OptimalTreeLearner, constraints::Array{BlackBoxFn}, X;
                                                jump_model::Union{JuMP.Model, Nothing} = nothing,
                                                weights=:autobalance,
                                                validation_criterion=:misclassification,
                                                return_samples::Bool=false)
    """
    Returns a set of feasibility trees from a set of constraints.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        X: initial matrix of feature data
        constraints: set of BlackBoxFns in std form (>= 0)
    Returns:
        lnr: list of Fitted Grids
    NOTE: All constraints must take in full vector of X values.
    """
    n_samples, n_features = size(X);
    n_constraints = length(constraints);
    Y = hcat([vcat([constraints[i](X[j, :]) >= 0 for j = 1:n_samples]...) for i=1:n_constraints]...);
    feas = [sum(Y[:,i]) > 0 for i= 1:n_constraints];
    if any(feas .== 0)
        @info("Certain constraints are infeasible for all samples.")
        @info("Will resample as necessary.")
    end
    grids = [gridify(lnr) for _ = 1:n_constraints];
    # First train the feasible trees...
    for i in findall(feas)
        grids[i] = learn_from_data!(X, Y[:,i], grids[i], idxs = constraints[i].idxs,
                                               weights=weights,
                                               validation_criterion=:misclassification);
    end
    if return_samples
        return grids, X
    else
        return grids
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

# function fit!(bbf::BlackBoxFn; lnr::IAI.Learner=base_otc(),
#                             weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
#                             validation_criterion::Symbol = :misclassification,
#                             return_samples = false)
#     """ Fits a provided function model with feasibility and obj f'n fits and
#         saves the learners.
#     """
#     n_samples, n_features = size(bbf.samples)
#     trees, X = learn_constraints!(lnr, md.fns, X, weights = weights,
#                                     validation_criterion=validation_criterion,
#                                     return_samples=true);
#
#     if dir != "-"
#         for i=1:size(trees, 1)
#             IAI.write_json(string(dir, "_tree_", i, ".json"),
#                            trees[i]);
#         end
#     end
#     return trees
# end

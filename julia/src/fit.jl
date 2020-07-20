using Gurobi
using JuMP

include("constraintify.jl")
include("model_data.jl")

function base_otr()
    return IAI.OptimalTreeRegressor(
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6,
        minbucket = 0.03,
        regression_sparsity = :all,
        fast_num_support_restarts = 5,
        hyperplane_config = (sparsity = :all,),
        regression_lambda = 0.00001,
        regression_weighted_betas = true,
    )
end

function base_otc()
    return IAI.OptimalTreeClassifier(
        random_seed = 1,
        max_depth = 5,
        cp = 1e-6,
        minbucket = 0.03,
        fast_num_support_restarts = 5,
        hyperplane_config = (sparsity = :all,),
    )
end

function base_grid(lnr)
    grid = IAI.GridSearch(lnr, Dict(:criterion => [:entropy, :misclassification],
    :normalize_X => [true],
    :max_depth => [3, 5],
    :minbucket => [0.3, 0.5]))
    return grid
end

function gridify(lnr::IAI.Learner)
    """ Turns IAI.Learners into IAI.GridSearches."""
    if !hasproperty(lnr, :lnr)
        grid = IAI.GridSearch(lnr);
    else
        grid = lnr;
    end
    return grid
end

function regress(Y, X, rho, p, M=2.)
    m = Model(solver=GurobiSolver())
    n_points = size(X, 1); # number of data points
    n_traits = size(X, 2); # number of traits
    @variable(m, b[i=1:n_traits])
    @variable(m, b0)
    @variable(m, t) # fitting error
    @variable(m, s) # sparsity penalty
    @constraint(m, sum((Y - X*b - b0).^2) <= t) # The fit cost
    if p == 0.
        @variable(m, z[1:n_traits], Bin)
        @constraint(m, scon, rho*sum(z[:]) <= s)
        @constraint(m, b[:] .<= M*z[:])
        @constraint(m, -b[:] .<= M*z[:])
    elseif p == 1.
        @variable(m, pb[1:n_traits])
        @constraint(m, pb[:] .>= b[:])
        @constraint(m, pb[:] .>= -b[:])
        @constraint(m, scon, rho*sum(pb) <= s)
    elseif p == 2.
        @constraint(m, scon, rho*sum(b.^2) <= s)
    else
        print("Error: this norm is not supported.")
    end
    @objective(m, Min, t + s);
    status = solve(m)
    return getvalue(b), getvalue(b0), getvalue(t), getvalue(s)
end

function learn_from_data!(X, Y, grids; idxs=nothing,
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
    n_samples, n_cols = size(Y);
    for i = 1:n_cols
        # Making sure that we only consider relevant features.
        if !isnothing(idxs)
            IAI.set_params!(grids[i].lnr, split_features = idxs[i])
            if typeof(grids[i].lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(grids[i].lnr, regression_features=idxs[i])
            end
        else
            IAI.set_params!(grids[i].lnr, split_features = :all)
            if typeof(grids[i].lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(grids[i].lnr, regression_features=:all)
            end
        end
        IAI.fit!(grids[i], X, Y[:,i],
                 validation_criterion = :misclassification, sample_weight=weights);
    end
    return grids
end

function learn_constraints(lnr::IAI.OptimalTreeLearner, constraints, X;
                                                jump_model::Union{JuMP.Model, Nothing} = nothing,
                                                idxs::Union{Array, Nothing} = nothing,
                                                weights=:autobalance,
                                                validation_criterion=:misclassification,
                                                return_samples::Bool=false)
    """
    Returns a set of feasibility trees from a set of constraints.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        X: initial matrix of feature data
        constraints: set of constraint functions in std form (>= 0)
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
    if any(feas .> 0)
        if isa(idxs, Nothing)
            grids[findall(feas)] = learn_from_data!(X, Y, grids,
                                               weights=weights, validation_criterion=:misclassification)
        else
            grids[findall(feas)] = learn_from_data!(X, Y, grids, idxs = idxs[findall(feas)],
                                               weights=weights, validation_criterion=:misclassification)
        end
    end
    if return_samples
        return grids, samples
    else
        return grids
    end
end

function fit(md::ModelData; X::Union{Array, Nothing} = nothing,
                            n_samples = 1000, jump_model::Union{JuMP.Model, Nothing} = nothing,
                            lnr::IAI.Learner=base_otc(),
                            weights::Union{Array, Symbol} = :autobalance, dir::String = "-",
                            validation_criterion::Symbol = :misclassification)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    if isa(X, Nothing)
        X = sample(md, n_samples=n_samples);
    end
    n_features = length(md.c);
    ineq_trees = learn_constraints(lnr, md.ineq_fns, X, idxs = md.ineq_idxs, weights = weights,
                                    validation_criterion=validation_criterion);
    eq_trees = learn_constraints(lnr, md.eq_fns, X, idxs = md.eq_idxs, weights = weights,
                                  validation_criterion=validation_criterion);
    if dir != "-"
        for i=1:size(ineq_trees,1)
            IAI.write_json(string(dir, "_ineq_", i, ".json"),
                           ineq_trees[i]);
        end
        for i=1:size(eq_trees,1)
            IAI.write_json(string(dir, "_eq_", i, ".json"),
                           eq_trees[i]);
        end
    end
    return ineq_trees, eq_trees
end


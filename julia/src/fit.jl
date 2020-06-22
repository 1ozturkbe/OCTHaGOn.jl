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
    :max_depth => [3, 4, 5],
    :minbucket => [0.3, 0.5]))
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

function learn_constraints(lnr, constraints, X; idxs = nothing, weights=ones(size(X,1)),
                                                 validation_criterion=:misclassification)
    """
    Returns a set of feasibility trees from a set of constraints.
    Arguments:
        lnr: Unfit OptimalTreeClassifier or Grid
        constraints: set of constraint functions in std form (>= 0)
        X: samples of the free variables
    Returns:
        lnr: list of Fitted Grids
    NOTE: All constraints must take in full vector of X values.
    """
    n_samples, n_features = size(X);
    n_constraints = length(constraints);
    feasTrees = IAI.GridSearch[];
    if !hasproperty(lnr, :lnr)
        grid = IAI.GridSearch(lnr);
    else
        grid = lnr;
    end
    for i = 1:n_constraints
        Y = vcat([constraints[i](X[j, :]) >= 0 for j = 1:n_samples]...);
        # Making sure that we only consider relevant features.
        if !isnothing(idxs)
            IAI.set_params!(grid.lnr, split_features = idxs[i])
            if typeof(grid.lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(grid.lnr, regression_features=idxs[i])
            end
        else
            IAI.set_params!(grid.lnr, split_features = :all)
            if typeof(grid.lnr) == IAI.OptimalTreeRegressor
                IAI.set_params!(grid.lnr, regression_features=:all)
            end
        end
        IAI.fit!(grid, X, Y, validation_criterion = :misclassification, sample_weight=weights)
        append!(feasTrees, [grid])
    end
    return feasTrees
end

function learn_objective!(lnr, objective, X; idxs=nothing, lse=false, weights=ones(size(X,1)))
    """
    Returns a set of feasibility trees from a set of constraints.
    Arguments:
        lnr: Unfit OptimalTreeClassifier, OptimalTreeRegressor, or Grid
        constraints: set of constraint functions in std form (>= 0)
        X: samples of the free variables
    NOTE: All constraints must take in full vector of X values.
    """
    n_samples, n_features = size(X)
    Y = vcat([objective(X[j, :]) for j = 1:n_samples]...);
    Y = Y[randperm(length(Y))]
    if !hasproperty(lnr, :lnr)
        grid = IAI.GridSearch(lnr);
    else
        grid = lnr;
    end
    if typeof(grid.lnr) == IAI.OptimalTreeClassifier
        nX = deepcopy(X);
        if lse && !any(Y .<= 0)
            nX = [nX log.(Y)];
        elseif lse
            print("WARNING: [Cannot fit in logspace, since objective " *
            "has negative values. LSE option is turned off.]")
            lse = false;
            nX = [nX Y];
        else
            nX = [nX Y];
        end
        nY = [Y[j] >= objective(nX[j,:]) for j=1:size(Y,1)];
    else
        nX = X;
        if lse && !any(Y .<= 0)
            nY = log.(Y);
        elseif lse
            print("WARNING: [Cannot fit in logspace, since objective " *
            "has negative values. LSE option is turned off.]")
            nY = Y;
            lse = false;
        else
            nY = Y;
        end
    end
    # Making sure that we only consider relevant features.
    if !isnothing(idxs)
        if typeof(grid.lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(grid.lnr, split_features = idxs)
            IAI.set_params!(grid.lnr, regression_features = idxs)
        else
            nidxs = append!(copy(idxs), n_features+1)
            IAI.set_params!(grid.lnr, split_features = nidxs)
        end
    else
        IAI.set_params!(grid.lnr, split_features = :all)
        if typeof(grid.lnr) == IAI.OptimalTreeRegressor
            IAI.set_params!(grid.lnr, regression_features=:all)
        end
    end
    IAI.fit!(grid, nX, nY, validation_criterion = :misclassification, sample_weight=weights)
    return grid
end

function fit(md::ModelData, X; lnr=base_otc(), weights=ones(size(X,1)), dir="-",
                               validation_criterion=:misclassification)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    n_samples, n_features = size(X);
    @assert n_features == length(md.c);
    n_dims = length(md.lbs);
    weights = ones(n_samples)
    ineq_trees = learn_constraints(lnr, md.ineq_fns, X, idxs = md.ineq_idxs, weights = weights,
                                    validation_criterion=validation_criterion)
    eq_trees = learn_constraints(lnr, md.eq_fns, X, idxs = md.eq_idxs, weights = weights,
                                  validation_criterion=validation_criterion)
    if dir != "-"
        for i=1:size(ineq_trees,1)
            IAI.write_json(string(dir, "_ineq_", i, ".json"),
                           ineq_trees[i])
        end
        for i=1:size(eq_trees,1)
            IAI.write_json(string(dir, "_eq_", i, ".json"),
                           eq_trees[i])
        end
    end
    return ineq_trees, eq_trees
end


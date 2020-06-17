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
    grid = IAI.GridSearch(lnr, Dict(:criterion => [:gini, :misclassification],
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

function example_fit(md::ModelData; lnr=base_otc())
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    n_samples = 1000;
    n_dims = length(md.lbs);
    weights = ones(n_samples)
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
    ineq_trees = learn_constraints!(md, md.ineq_fns, X, weights=weights, lnr=lnr)
    for i=1:size(ineq_trees,1)
        IAI.write_json(string("data/", md.name, "_ineq_", i, ".json"),
                       ineq_trees[i])
    end
    return true
end
using Gurobi
using JuMP
# Defining regression function, with rho and pnorm

function base_otr()
    return IAI.OptimalTreeRegressor(
        random_seed = 1,
        max_depth = 3,
        cp = 1e-6,
        minbucket = 0.03,
        regression_sparsity = :all,
        fast_num_support_restarts = 1,
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
        minbucket = 0.01,
        fast_num_support_restarts = 1,
        hyperplane_config = (sparsity = :all,),
    )
end

function fit_fn_model(fn_model, X; weights=ones(size(X,1)))
    ineq_trees = learn_constraints(base_otc(), fn_model.ineqs, X; idxs=fn_model.ineq_idxs,
                                   weights=weights)
    eq_trees = learn_constraints(base_otc(), fn_model.eqs, X; idxs=fn_model.eq_idxs,
                                 weights=weights)
    obj_tree = learn_objective!(base_otc(), fn_model.obj, X;
                               idxs=fn_model.obj_idxs, lse=fn_model.lse,
                               weights=weights)
    return obj_tree, ineq_trees, eq_trees
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

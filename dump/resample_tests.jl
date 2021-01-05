using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test
using Plots

"""
Set of examples for which to test different examples.
"""

function resample_test(fn_model)
    n_samples = 1000;
    n_dims = length(fn_model.lbs);
    n_iterations = 4;
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    global X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    for i=1:n_iterations
        obj_tree, ineq_trees = fit_fn_model(fn_model, X, weights=weights);
        # Creating JuMP model
        m = Model();
        set_optimizer(m, Gurobi.Optimizer)
        n_vars = length(fn_model.lbs);
        vks = [Symbol("x",i) for i=1:n_vars];
        obj_vks = [Symbol("x",i) for i=1:n_vars+1];
        @variable(m, x[1:n_vars]);
        @variable(m, y);
        @objective(m, Min, y);
        add_feas_constraints!(obj_tree, m, [x; y], obj_vks; M = 1e2);
        for tree in ineq_trees
            add_feas_constraints!(tree, m, x, vks; M = 1e2);
        end
        for tree in eq_trees
            add_feas_constraints!(tree, m, x, vks; M = 1e2);
        end
        for i=1:n_vars # Bounding
            @constraint(m, x[i] <= fn_model.ubs[i]);
            @constraint(m, x[i] >= fn_model.lbs[i]);
        end
        status = optimize!(m)
        println("X values: ", getvalue.(x))
        println("Objective cost: ", getvalue.(y))

        global X = resample(n_samples, fn_model.lbs, fn_model.ubs, X, getvalue.(x);
                     mean_shift=0.5, std_shrink = 0.3)
    end
    # Some post processing
    # for i=1:n_iterations
    #     # fig = scatter(X[1+(i-1)*n_samples: i*n_samples, 1], X[1+(i-1)*n_samples: i*n_samples, 2],
    #     fig = scatter(X[1: i*n_samples, 1], X[1: i*n_samples, 2],
    #     xlim=[fn_model.lbs[1],fn_model.ubs[1]], ylim=[fn_model.lbs[2],fn_model.ubs[2]]);
    #     savefig(fig, "figures/"*fn_model.name*"_plot_"*string(i))
    # end
    return true
end

@test test_sagemark_to_GlobalModel()
# fn_model = sagemark_to_GlobalModel(3, lse=true);
# @test example_fit(fn_model, lnr=base_lnr(false))
# @test example_solve(fn_model, M=1e5)

# Importing MINLP cases
# using MINLPLib
# m = fetch_model("minlp2/blend029")
# using Cbc, Juniper
# ipopt = IpoptSolver(print_level=1)
# gurobi = GurobiSolver()
# juniper = JuniperSolver(ipopt, mip_solver=gurobi)
# setsolver(m, juniper)
# solve(m)
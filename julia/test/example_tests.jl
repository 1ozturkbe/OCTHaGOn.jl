using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test
using Plots

include("../src/tools.jl")
include("../src/fit.jl")
include("../src/constraintify.jl")
include("../src/solve.jl")
include("../src/post_process.jl")

"""
Set of examples for which to test different examples.
"""

function test_sagemark_to_ModelData()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = sagemark_to_ModelData(idx);
    end
    idx = 1;
    ex = sagemark_to_ModelData(idx; lse=false);
    ex_lse = sagemark_to_ModelData(idx; lse=true);
    ubs = [1, 10, 15, 1];
    lbs = [0.1, 5, 8, 0.01];
    inp = [1,1.9,3,3.9];
    @test ex.obj(inp) ≈ inp[3] ^ 0.8 * inp[4] ^ 1.2
    @test ex_lse.obj(log.(inp)) ≈ inp[3] ^ 0.8 * inp[4] ^ 1.2
    for i=1:length(inp)
        @test ex.ubs[i] ≈ ubs[i];
        @test ex.lbs[i] ≈ lbs[i];
        @test ex_lse.ubs[i] ≈ log(ubs[i])
        @test ex_lse.lbs[i] ≈ log(lbs[i])
    end
    @test ex.ineqs[1](inp) >= 0
    @test ex.ineqs[2](inp) <= 0
    @test ex.ineqs[1](inp) ≈ ex_lse.ineqs[1](log.(inp))
    return true
end

function resample_test(fn_model)
    n_samples = 1000;
    n_dims = length(fn_model.lbs);
    n_iterations = 4;
    weights = ones(n_samples);
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    global X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    for i=1:n_iterations
        obj_tree, ineq_trees = fit_fn_model(fn_model, X, weights=weights);
        # Creating JuMP model
        m = Model(solver=GurobiSolver());
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
        status = solve(m)
        println("X values: ", getvalue(x))
        println("Objective cost: ", getvalue(y))

        global X = resample(n_samples, fn_model.lbs, fn_model.ubs, X, getvalue(x);
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

@test test_sagemark_to_ModelData()
fn_model = sagemark_to_ModelData(3, lse=true);
@test example_fit(fn_model, lnr=base_otc())
@test example_solve(fn_model, M=1e5)

# Importing MINLP cases
# using MINLPLib
# m = fetch_model("minlp2/blend029")
# using Cbc, Juniper
# ipopt = IpoptSolver(print_level=1)
# gurobi = GurobiSolver()
# juniper = JuniperSolver(ipopt, mip_solver=gurobi)
# setsolver(m, juniper)
# solve(m)

# # Fitting the ModelData
# n_samples = 1000;
# n_dims = length(md.c);
# fn_model.lbs = [maximum([fn_model.lbs[i], mof_vars[i]-40]) for i=1:n_dims]
# fn_model.ubs = [minimum([fn_model.ubs[i], mof_vars[i]+40]) for i=1:n_dims]
# example_fit(fn_model, lnr=base_otc())
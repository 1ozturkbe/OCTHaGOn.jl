using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test

include("examples.jl")
include("../src/constraintify.jl")
include("../src/solve.jl")

"""
Set of examples for which to test different examples.
"""

function example_fit(fn_model::function_model)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    n_samples = 1000;
    n_dims = length(fn_model.lbs);
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    otr = base_otr();
    otc = base_otc();
    ineq_trees = learn_constraints(otc, fn_model.ineqs, X; idxs=fn_model.ineq_idxs);
    eq_trees = learn_constraints(otc, fn_model.eqs, X; idxs=fn_model.eq_idxs);
    obj_tree = learn_objective(otr, fn_model.obj, X;
                               idxs=fn_model.obj_idxs, lse=fn_model.lse)
    IAI.write_json(string("data/", fn_model.name, "_obj.json"), obj_tree);
    for i=1:size(ineq_trees,1)
        IAI.write_json(string("data/", fn_model.name, "_ineq_", i, ".json"),
                       ineq_trees[i]);
    end
    for i = 1:size(eq_trees, 1)
        IAI.write_json(string("data/", fn_model.name, "_eq_", i, ".json"),
                       eq_trees[i])
    end
end

function example_solve(fn_model::function_model)
    """ Solves an already fitted function_model. """
    # Retrieving constraints
    try
        obj_tree = IAI.read_json(string("data/", fn_model.name, "_obj.json"))
        ineq_trees = [IAI.read_json(string("data/", fn_model.name, "_ineq_", i, ".json")) for i=1:length(fn_model.ineqs)];
        eq_trees = [IAI.read_json(string("data/", fn_model.name, "_eq_", i, ".json")) for i=1:length(fn_model.eqs)];
    catch
        print("You have not yet fitted the components of ", fn_model.name, '.')
    end
    # Creating JuMP model
    m = Model(solver=GurobiSolver());
    n_vars = length(fn_model.lbs);
    vks = [Symbol("x",i) for i=1:n_vars];
    @variable(m, x[1:n_vars])
    @variable(m, y)
    @objective(m, Min, y)
    add_mio_constraints(obj_tree, m, x, y, vks; M = 1e2)
    for tree in ineq_trees
        add_feas_constraints(tree, m, x, vks; M = 1e2)
    end
    for tree in eq_trees
        add_feas_constraints(tree, m, x, vks; M = 1e2)
    end
    bound_variables(m, x, fn_model.lbs, fn_model.ubs);
    status = solve(m)
    print(status)
    return true
end


function test_import_sagebenchmark()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = import_sagebenchmark(idx);
    end
    idx = 1;
    ex = import_sagebenchmark(idx; lse=false);
    ex_lse = import_sagebenchmark(idx; lse=true);
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

@test test_import_sagebenchmark()

fn_model = import_sagebenchmark(1, lse=true);

# @test example_fit(fn_model)
@test example_solve(fn_model)

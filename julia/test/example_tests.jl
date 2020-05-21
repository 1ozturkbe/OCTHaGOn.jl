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

function example_constraint_fit(fn_model)
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    n_samples = 1000;
    n_dims = length(fn_model.lbs)
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    otr = base_otr()
    otc = base_otc()
    ineq_trees = learn_constraints(otc, fn_model.ineqs, X; idxs=fn_model.ineq_idxs)
    eq_trees = learn_constraints(otc, fn_model.eqs, X; idxs=fn_model.eq_idxs)
    for i=1:size(ineq_trees,1)
        IAI.write_json(lnr, "data/" + name + "_ineq_" + string(i) + ".json")
    end
    for i = 1:size(eq_trees, 1)
        IAI.write_json(lnr, "data/" + name + "_eq_" + string(i) + ".json")
    end
end

function example_naive_fit(fn_model)
    """
    Fits a provided function_model.
    Arguments:
        fn_model:: instance of function_model
    """
    n_samples = 1000;
    n_dims = 3;
    # Code for initial tree training
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:3]);
    lnr = base_otr()
    for i = 1:size(fn_model.constr, 1)
        Y = [-fn_model.constr[i](X[j,:]) for j=1:n_samples];
        IAI.fit!(lnr, X, Y)
        IAI.write_json(lnr, "data/" + name + "constraint" + string(i) + "_reg.json")
    end
    Y = [fn_model.obj(X[j,:]) for j=1:n_samples];
    IAI.fit!(lnr, X, Y)
    IAI.write_json("data/" + name + "_objective.json", lnr)
end

# function example_naive_solve(fn_model)
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_naive.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(solver=GurobiSolver());
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_mio_constraints(constr, m, x, 0, vks, 1000000);
#     add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
#     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
#     status = solve(m)
#     println("Solved minimum: ", getvalue(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue(x))
#     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
# end

# function example1_infeas()
#     n_samples = 1000;
#     n_dims = 3;
#     # Code for initial tree training
#     plan, _ = LHCoptim(n_samples, n_dims, 1);
#     X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:3]);
#     # Assuming the objective has already been trained...
#     lnr = base_otr()
#     name = "example1"
#     feasTrees = learn_constraints(lnr, constraints, X, name=name)
#
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_infeas.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(solver=GurobiSolver());
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_feas_constraints(constr, m, x, vks, 1000);
#     add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
#     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
#     status = solve(m)
#     println("Solved minimum: ", getvalue(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue(x))
#     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
# end



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

# @test example_

# @test example_naive_fit(example2)

# @test example1_infeas()
# fn_model = import_sagebenchmark(1, lse=true);
# n_samples = 1000;
# n_dims = length(fn_model.lbs)
# vks = [Symbol("x",i) for i=1:n_dims];
# plan, _ = LHCoptim(n_samples, n_dims, 1);
# X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
# otr = base_otr()
# otc = base_otc()
# # obj_tree = learn_objective(otc, fn_model.obj, X; idxs=fn_model.obj_idxs)
# ineq_trees = learn_constraints(otc, fn_model.ineqs, X; idxs=fn_model.ineq_idxs)
# eq_trees = learn_constraints(otc, fn_model.eqs, X; idxs=fn_model.eq_idxs)
# for i=1:size(ineq_trees,1)
#     IAI.write_json(string("data/", fn_model.name, "_ineq_", string(i), ".json"),
#                     ineq_trees[i]);
# end
# for i = 1:size(eq_trees, 1)
#     IAI.write_json(string("data/", fn_model.name, "_eq_", string(i), ".json"),
#                    eq_trees[i]);
# end
# m = Model(solver=GurobiSolver());
# @variable(m, x[1:n_dims])
# @variable(m, obj)
# @objective(m, Min, obj)
# for i=1:length(ineq_trees)
#     add_feas_constraints(ineq_trees[i], m, x, vks; M = 100);
# end
# for i=1:length(eq_trees)
#     add_feas_constraints(eq_trees[i], m, x, vks; M = 100);
# end
#
# #     add_mio_constraints(constr, m, x, 0, vks, 1000000);
# #     add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
# #     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
# #     status = solve(m)
# #     println("Solved minimum: ", getvalue(obj))
# #     println("Known global bound: ", -147-2/3)
# #     println("X values: ", getvalue(x))
# #     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
#
# n_samples, n_features = size(X)
# objective = fn_model.obj
# Y = [objective(transpose(X[j, :])) for j = 1:n_samples]
# nX = repeat(X, n_samples)
# nY = repeat(Y, n_samples)
# nY = [nY[j] >= objective(nX[j,:]) for j=1:shape(nY,2)]
#     # Making sure that we only consider relevant features.
#     if !isnothing(idxs)
#         IAI.set_params!(lnr, split_features = idxs[i])
#         if typeof(lnr) == IAI.OptimalTreeRegressor
#             IAI.set_params!(lnr, regression_features=idxs[i])
#         end
#     else
#         IAI.set_params!(lnr, split_features = :all)
#         if typeof(lnr) == IAI.OptimalTreeRegressor
#             IAI.set_params!(lnr, regression_features=:all)
#         end
#     end
#     IAI.fit!(lnr, nX, nY)
#     return lnr

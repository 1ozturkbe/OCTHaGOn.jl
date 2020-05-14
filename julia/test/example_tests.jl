using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test

include("examples.jl")
include("../gen_constraints.jl")

"""
Set of examples for which to test different examples.
"""


function example_naive_fit(fnm)
    """
    Fits a provided function_model.
    Arguments:
        fnm:: instance of function_model
    """
    nsamples = 1000;
    ndims = 3;
    # Code for initial tree training
    plan, _ = LHCoptim(nsamples, ndims, 1);
    X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
    lnr = IAI.OptimalTreeRegressor(random_seed=1, max_depth=3, cp=1e-10,  minbucket=0.03, regression_sparsity=:all, fast_num_support_restarts = 1, hyperplane_config=(sparsity=:1,), regression_lambda = 0.00001, regression_weighted_betas=true)
    for i = 1:size(fnm.constr, 1)
        Y = [-fnm.constr[i](X[j,:]) for j=1:nsamples];
        IAI.fit!(lnr, X, Y)
        IAI.write_json(lnr, "data/" + name + "constraint" + string(i) + "_reg.json")
    end
    Y = [fnm.obj(X[j,:]) for j=1:nsamples];
    IAI.fit!(lnr, X, Y)
    IAI.write_json("data/" + name + "_objective.json", lnr)
end

function example_naive_solve(fnm)
    # Creating the model
    constr = IAI.read_json("data/example1_constraint_naive.json")
    objectivefn = IAI.read_json("data/example1_objective_naive.json")
    vks = [Symbol("x",i) for i=1:3];
    m = Model(solver=GurobiSolver());
    @variable(m, x[1:3])
    @variable(m, obj)
    @objective(m, Min, obj)
    add_mio_constraints(constr, m, x, 0, vks, 1000000);
    add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
    constraints_from_bounds(m, x, fnm.lbs, fnm.ubs);
    status = solve(m)
    println("Solved minimum: ", getvalue(obj))
    println("Known global bound: ", -147-2/3)
    println("X values: ", getvalue(x))
    println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
end

function example1_infeas()
    nsamples = 1000;
    ndims = 3;
    # Code for initial tree training
    plan, _ = LHCoptim(nsamples, ndims, 1);
    X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
    # Assuming the objective has already been trained...
    lnr = IAI.OptimalTreeClassifier(random_seed=1, max_depth=5, cp=1e-10,  minbucket=0.01,fast_num_support_restarts = 1, hyperplane_config=(sparsity=1,))
    name = "example1"
    feasTrees = learn_constraints(lnr, constraints, X, name=name)

    # Creating the model
    constr = IAI.read_json("data/example1_constraint_infeas.json")
    objectivefn = IAI.read_json("data/example1_objective_naive.json")
    vks = [Symbol("x",i) for i=1:3];
    m = Model(solver=GurobiSolver());
    @variable(m, x[1:3])
    @variable(m, obj)
    @objective(m, Min, obj)
    add_feas_constraints(constr, m, x, vks, 1000);
    add_mio_constraints(objectivefn, m, x, obj, vks, 1000000);
    constraints_from_bounds(m, x, fnm.lbs, fnm.ubs);
    status = solve(m)
    println("Solved minimum: ", getvalue(obj))
    println("Known global bound: ", -147-2/3)
    println("X values: ", getvalue(x))
    println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
end

function model_creation()
    examples = [example1, example2]
    for i in examples
        ex = i()
    end
    return true
end

@test model_creation()

@test example_naive_fit(example2)
    

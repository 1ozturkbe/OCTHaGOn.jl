using DataFrames
using Gurobi, JuMP
using LatinHypercubeSampling
using Test

include("../gen_constraints.jl")

function example1()
    objective(x) = 0.5*exp(x[1]-x[2]) - exp(x[1]) - 5*exp(-x[2])
    g1(x) = 100 - exp(x[2]-x[3]) - exp(x[2]) - 0.05*exp(x[1]+x[3])
    lbs = log.([70, 1, 0.5])
    ubs =  log.([150, 20, 21])
    nsamples = 1000
    ndims = 3
    # Code for initial tree training
    #     plan, _ = LHCoptim(nsamples, ndims, 1)
    #     X = scaleLHC(plan,[(lbs[i], ubs[i]) for i=1:3])   
    #     lnr = IAI.OptimalTreeRegressor(random_seed=1, max_depth=5, cp=1e-8,  minbucket=0.05, regression_sparsity=:all, fast_num_support_restarts = 1,
    #                                    hyperplane_config=(sparsity=1,), regression_lambda = 0.0001)
    #     Y = [-g1(X[j,:]) for j=1:nsamples]
    #     IAI.fit!(lnr, X, Y)
    #     IAI.write_json("data/example1_constraint.json", lnr)
    #     Y = [objective(X[j,:]) for j=1:nsamples]
    #     IAI.fit!(lnr, X, Y)
    #     IAI.write_json("data/example1_objective.json", lnr)
    
    # Creating the model
    constr = IAI.read_json("data/example1_constraint.json")
    objectivefn = IAI.read_json("data/example1_objective.json")
    vks = [Symbol("x",i) for i=1:3];
    m = Model(solver=GurobiSolver());
    @variable(m, x[1:3])
    @variable(m, y)
    @variable(m, obj)
    @objective(m, Min, obj)
    add_mio_constraints(constr, m, x, 0, vks, 1000);
    add_mio_constraints(objectivefn, m, x, obj, vks, 1000);
    constraints_from_bounds(m, x, lbs, ubs);
    status = solve(m)
    println("Solved minimum: ", getvalue(obj))
    println("Known global bound: ", 0.00371)
    println("X values: ", getvalue(x))
    println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
    return true
end

@test example1()

    
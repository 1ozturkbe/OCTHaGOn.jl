include("constraintify.jl")
include("../test/examples.jl")

function reweight(X, x_sol, mag=2, neighbors=1.2)
    """ Reweights existing data by proximity to previous solution.
    Arguments:
    - X: data
    - x_sol: solution array with same # of columns
    - mag: relative weighting, bigger is more.
    - neighbors: magnitude of relative deviation where weighting is equal.
    Returns:
    - weights: weights of X rows, by Euclidian distance
    """
    distance = 0;
    weights =[];
    return weights
end

function solve_sagebenchmark(sagemark)
    """ Solves a benchmark of given number (1-25 for now). """
    nsamples = 1000;
    ndims = 3;
    # Code for initial tree training
    plan, _ = LHCoptim(nsamples, ndims, 1);
    X = scaleLHC(plan,[(fnm.lbs[i], fnm.ubs[i]) for i=1:3]);
    # Assuming the objective has already been trained...
    otr = base_otr()
    otc = base_otc()
    name = "example1"
    feasTrees = learn_constraints(otc, constraints, X, name=name)

    # Creating the model
    constr = IAI.read_json("data/example1_constraint_infeas.json")
    objectivefn = IAI.read_json("data/example1_objective_naive.json")
    vks = [Symbol("x",i) for i=1:3];
    m = Model(solver=GurobiSolver());
    @variable(m, x[1:3])
    @variable(m, obj)
    @objective(m, Min, obj)
    add_feas_constraints!(constr, m, x, vks, 1000);
    add_mio_constraints!(objectivefn, m, x, obj, vks, 1000000);
    bound_variables!(m, x, fnm.lbs, fnm.ubs);
    status = solve(m)
    println("Solved minimum: ", getvalue(obj))
    println("Known global bound: ", -147-2/3)
    println("X values: ", getvalue(x))
    println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
end

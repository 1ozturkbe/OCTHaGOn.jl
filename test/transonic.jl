using CSV, DataFrames, Statistics, Random
using Gurobi, JuMP
using Test

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree
global PROJECT_ROOT = @__DIR__

function train_transonic_tree()
    @info "Trains tree over transonic airfoil data, testing IAI ORT fitting,
           basic tree data structures, and ORT constraint generation."
    # Getting data from aerospace database.
    vks = ["Re", "thick", "M", "C_L"];
    X = CSV.read("../data/airfoil/airfoil_X.csv",
                 copycols=true, header=vks, delim=",");
    Y = CSV.read("../data/airfoil/airfoil_y.csv",
                 copycols=true, header=["C_D"], delim=",");
    (train_X, train_y), (test_X, test_y) = IAI.split_data(:regression, Matrix(X), Matrix(Y), train_proportion=.999);

    # Training tree
    lnr = OCT.base_otr()
    IAI.set_params!(lnr, hyperplane_config=(sparsity=1,))
    lnr = OCT.gridify(lnr);
    IAI.fit!(lnr, train_X, train_y)
    IAI.write_json(string(PROJECT_ROOT, "/data/transonic_tree.json"), lnr)
    return true
end

function transonic_mio_model()
    @info "Generates and solves mio model over transonic airfoil data."
    lnr = IAI.read_json(string(PROJECT_ROOT, "/data/transonic_tree.json"));
    Re = Array(range(10000, stop=35000, step=5000));
    thick = [0.100,0.110,0.120,0.130,0.140,0.145];
    M = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    cl = Array(range(0.35, stop=0.70, step=0.05));
    vks = [Symbol("x",i) for i=1:4];
    pwlDict = OCT.pwl_constraint_data(lnr.lnr, vks);
    upperDict, lowerDict = OCT.trust_region_data(lnr.lnr, vks);
    # Generate MIO constraints from aerodynamics data
    m = Model()
    set_optimizer(m, GUROBI_SILENT);
    @variable(m, x[1:4])
    @variable(m, y)
    @constraint(m, log(minimum(Re)) <= x[1])
    @constraint(m, x[1] <= log(maximum(Re)))
    @constraint(m, log(minimum(thick)) <= x[2])
    @constraint(m, x[2] <= log(maximum(thick)))
    @constraint(m, log(minimum(M)) <= x[3])
    @constraint(m, x[3] <= log(maximum(M)))
    @constraint(m, log(minimum(cl)) <= x[4])
    @constraint(m, x[4] <= log(maximum(cl)))
    OCT.add_regr_constraints!(m, x, y, lnr, vks, M=100., eq=true);
    # Setting optimization constraints (as a demonstration)
    # Re, thickness, M, cl
    @constraint(m, x[1] >= log(20000))
    @constraint(m, x[2] >= log(0.11))
    @constraint(m, x[3] >= log(0.7))
    @constraint(m, x[4] >= log(0.4))
    @objective(m, Min, y)
    status = optimize!(m)
    println("Actual Cd: ", exp(getvalue.(y)))
    println("Min Cd in data: ", 0.00371)
    return true
end

@test train_transonic_tree()

@test transonic_mio_model()

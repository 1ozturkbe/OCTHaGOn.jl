#=
test_knapsack:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
=#

using Test

using Distributions
using ExperimentalDesign
using Gurobi
using JuMP
using Random
Random.seed!(1);
include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;

function knapsack_fn(a,b,x)
    return x -> sum(a.*x) <= b
end

# Example knapsack
n = 10;
a = rand(n);
c = rand(n);
b = 10/3*rand();

# Let's find the JuMP solution
m = Model(solver=GurobiSolver());
@variable(m, x[1:n], Bin);
@constraint(m, sum(a.*x) <= b);
@objective(m, Max, sum(c.*x));
status = solve(m);

# Let's try to solve via a tree.

# X = ExperimentalDesign.FullFactorial(tuple([0,1] for i=1:n))
# X = explicit_fullfactorial(tuple(fill([0,1],n)))
pb = PlackettBurman(n);

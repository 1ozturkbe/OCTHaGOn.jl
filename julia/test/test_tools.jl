#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests src/tools.jl.
=#

using Test
using MathOptInterface;
using JuMP;
global MOI = MathOptInterface;
MOI.Silent() = true
include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
const PROJECT_ROOT = @__DIR__

function test_sagemark_to_ModelData()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = OCT.sagemark_to_ModelData(idx);
    end
    idx = 1;
    md = OCT.sagemark_to_ModelData(idx; lse=false);
    md_lse = OCT.sagemark_to_ModelData(idx; lse=true);
    ubs = [1, 10, 15, 1, Inf];
    lbs = [0.1, 5, 8, 0.01, 1e-20];
    inp = [1,1.9,3,3.9, 10];
#     @test md.obj(inp) ≈ inp[3] ^ 0.8 * inp[4] ^ 1.2
#     @test md_lse.obj(log.(inp)) ≈ inp[3] ^ 0.8 * inp[4] ^ 1.2
    for i=1:length(inp)-1
        @test md.ubs[i] ≈ ubs[i];
        @test md.lbs[i] ≈ lbs[i];
        @test md_lse.ubs[i] ≈ log(ubs[i])
        @test md_lse.lbs[i] ≈ log(lbs[i])
    end
    @test md.ineq_fns[2](inp) >= 0
    @test md.ineq_fns[3](inp) <= 0
    @test md.ineq_fns[2](inp) ≈ md_lse.ineq_fns[2](log.(inp))
    return true
end

# Importing sagebenchmark to ModelData and checking it
@test test_sagemark_to_ModelData()
md = OCT.sagemark_to_ModelData(3, lse=false);
md.lbs[end] = -300;
md.ubs[end]= -0;
# Sampling ModelData, creating and solving a JuMP.Model
X = OCT.sample(md);
ineq_trees, eq_trees = OCT.fit(md, X, lnr = OCT.base_otc(),
                               dir=string("test/data/",md.name));
m, x = OCT.jump_it(md);
OCT.add_tree_constraints!(m, x, ineq_trees, eq_trees);
status = solve(m);
println("Solved minimum: ", sum(md.c .* getvalue(x)))
println("Known global bound: ", -147-2/3)
println("X values: ", getvalue(x))
println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))
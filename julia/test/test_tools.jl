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
global PROJECT_ROOT = @__DIR__

function test_sagemark_to_ModelData()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    @info("Testing sagemark imports.")
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = OCT.sagemark_to_ModelData(idx);
    end
    idx = 1;
    md = OCT.sagemark_to_ModelData(idx; lse=false);
    md_lse = OCT.sagemark_to_ModelData(idx; lse=true);
    ubs = Dict(md.vks .=> [1, 10, 15, 1, Inf]);
    lbs = Dict(md.vks .=> [0.1, 5, 8, 0.01, -Inf]);
    inp = Dict(md.vks .=> [1,1.9,3,3.9, 10]);
    log_inp = Dict(vk =>log(val) for (vk,val) in inp);
    @test md_lse.fns[1](log_inp) ≈ md.fns[1](inp) ≈ inp[:x5] - inp[:x3] ^ 0.8 * inp[:x4] ^ 1.2
    for i in md.vks
    println("Testing bounds of: ", i)
        @test md.ubs[i] ≈ ubs[i];
        @test md.lbs[i] ≈ lbs[i];
    end
    @test md.fns[2](inp) >= 0
    @test md.fns[3](inp) <= 0
    @test md.fns[2](inp) ≈ md_lse.fns[2](log_inp)
    return true
end

# Importing sagebenchmark to ModelData and checking it
@test test_sagemark_to_ModelData()
md = OCT.sagemark_to_ModelData(3, lse=false);
OCT.update_bounds!(md, lbs = Dict(:x4 => -300), ubs = Dict(:x4 => 0))

# Fitting all fns, and solving model
for fn in md.fns
    fn.n_samples = 1500;
    OCT.sample_and_eval!(fn);
    OCT.learn_constraint!(fn);
    println("Accuracy:", fn.accuracies[end])
end

OCT.jump_it!(md);
OCT.add_tree_constraints!(md);
status = solve(md.JuMP_model);
println("X values: ", getvalue(md.JuMP_vars))
println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))
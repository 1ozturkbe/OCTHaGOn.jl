#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests src/tools.jl.
=#

using PyCall

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
md = OCT.sagemark_to_ModelData(3, lse=true);
update_bounds!(md, lbs = Dict(:x4 => -300), ubs = Dict(:x4 => 0))

# Fitting all fns.
sample_and_eval!(md, n_samples=200)
learn_constraint!(md);
println("Approximation accuracies: ", accuracy(md))

# Solving the model.
add_tree_constraints!(md);
status = optimize!(md.model);
println("X values: ", solution(md))
println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

# Testing constraint addition and removal
clear_tree_constraints!(md) # Clears all BBF constraints
@test all([!is_valid(md.model, constraint) for constraint in md.fns[2].constraints])
md.fns[2].constraints, md.fns[2].leaf_variables = add_feas_constraints!(md.model, [md.vars[vk] for vk in md.fns[2].vks],
                                              md.fns[2].learners[end], md.fns[2].vks,
                                              return_data = true) # Adds only one bbf constraint
clear_tree_constraints!(md) # Finds and clears the one remaining BBF constraint.
@test all([!is_valid(md.model, constraint) for constraint in md.fns[1].constraints])
@test all([!is_valid(md.model, var) for var in md.fns[1].leaf_variables])


# Resampling and resolving via KNN
# sample_and_eval!(md);
# learn_constraint!(md);
# println("Approximation accuracies: ", accuracy(md))
#
# # Solving again
# globalsolve(md);
# println("X values: ", solution(md))
# println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))
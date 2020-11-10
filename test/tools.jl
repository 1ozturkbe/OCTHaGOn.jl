#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests src/tools.jl.
=#

using PyCall
using Ipopt

function test_sagemark_to_GlobalModel()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    @info("Testing sagemark imports.")
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = OCT.sagemark_to_GlobalModel(idx);
    end
    idx = 1;
    gm = OCT.sagemark_to_GlobalModel(idx; lse=false);
    gm_lse = OCT.sagemark_to_GlobalModel(idx; lse=true);
    classify_constraints.([gm, gm_lse])
    return true
end

# Importing sagebenchmark to ModelData and checking it
@test test_sagemark_to_GlobalModel()
gm = OCT.sagemark_to_GlobalModel(1, lse=false);
gm_lse = OCT.sagemark_to_GlobalModel(1, lse=true);
# set_optimizer(gm, Ipopt.Optimizer)
# nonlinearize!(gm)
# optimize!(gm)
# vals = getvalue.(gm.vars)

bounds = Dict(all_variables(gm) .=> [[0.1, 1], [5., 10.], [8., 15.], [0.01, 1.], [-Inf, Inf]])
@test all(get_bounds(gm)[var] ≈ bounds[var] for var in all_variables(gm))
inp = Dict(all_variables(gm) .=> [1,1.9,3,3.9, 10.]);
log_inp = Dict(vk => log(val) for (vk, val) in inp);

@test gm.vars == all_variables(gm);
@test gm_lse.bbfs[1](log_inp)[1] ≈ gm.bbfs[1](inp)[1] ≈ [inp[gm.vars[5]] - inp[gm.vars[3]] ^ 0.8 * inp[gm.vars[4]] ^ 1.2][1]

# Checking OCTException for sampling unbounded model
@test_throws OCTException sample_and_eval!(gm.bbfs[1], n_samples=200)

# Actually trying to optimize...
gm = OCT.sagemark_to_GlobalModel(3; lse=false);
set_optimizer(gm, Gurobi.Optimizer)
bound!(gm, Dict(gm.vars[end] => [-300, 0]))
sample_and_eval!(gm, n_samples = 500)
sample_and_eval!(gm, n_samples = 500)

learn_constraint!(gm)
println("Approximation accuracies: ", accuracy(gm))

# Testing addition and removal of tree constraints.
add_tree_constraints!(gm);
clear_tree_constraints!(gm, [gm.bbfs[1]])
@test !any(is_valid(gm.model, constraint) for constraint in gm.bbfs[1].mi_constraints)

# Solving of model
status = globalsolve(gm)
println("X values: ", solution(gm))
println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))


# Testing constraint addition and removal
# clear_tree_constraints!(md) # Clears all BBF constraints
# @test all([!is_valid(md.model, constraint) for constraint in md.bbfs[2].mi_constraints])
# md.bbfs[2].mi_constraints, md.bbfs[2].leaf_variables = add_feas_constraints!(md.model, [md.vars[vk] for vk in md.bbfs[2].vks],
#                                               md.bbfs[2].learners[end], md.bbfs[2].vks,
#                                               return_data = true) # Adds only one bbf constraint
# clear_tree_constraints!(md) # Finds and clears the one remaining BBF constraint.
# @test all([!is_valid(md.model, constraint) for constraint in md.bbfs[1].mi_constraints])
# @test all([!is_valid(md.model, var) for var in md.bbfs[1].leaf_variables])


# Resampling and resolving via KNN
# sample_and_eval!(md);
# learn_constraint!(md);
# println("Approximation accuracies: ", accuracy(md))
#
# # Solving again
# globalsolve(md);
# println("X values: ", solution(md))
# println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))
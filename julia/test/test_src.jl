#=
test_src:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code.
All the rest of the tests look at examples
=#

using Test

include("../src/OptimalConstraintTree.jl");
const OCT = OptimalConstraintTree;

# Initialization tests
md = OCT.ModelData(c = [1,2,3]);
@test md.lbs == -Inf.*ones(length(md.c))

# Check infeasible bounds
OCT.update_bounds!(md, [-2,1,3], [Inf, 5, 6]);
@test_throws ArgumentError OCT.update_bounds!(md, [-2,1,3], [Inf, -1, 6]);



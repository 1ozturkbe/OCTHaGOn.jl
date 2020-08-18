#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
Note: All tests should be run from a julia REPR within the julia folder, using:
      julia --project
      include("test/test_all.jl")
      To see coverage, run with:
      julia --project --code-coverage=tracefile-%p.info --code-coverage=user
      include("test/test_all.jl")
=#

using DataFrames
using JuMP
using MathOptInterface
using Test
using Random

include("../src/OptimalConstraintTree.jl")
using .OptimalConstraintTree
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__
Random.seed!(1);
MOI.Silent() = true

@testset "OptimalConstraintTree" begin
    include("test/test_bbf.jl")

    include("test/test_src.jl")

    include("test/test_tools.jl")

    include("test/test_sampling.jl")
end

# Other tests to try later

# include("test/test_transonic.jl");
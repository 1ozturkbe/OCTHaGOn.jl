#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
=#


include("../src/OptimalConstraintTree.jl")
using Test

@testset "OptimalConstraintTree" begin
    @test 1+1 == 2

    include("test_src.jl")

    include("test_tools.jl")

    include("test_knapsack.jl")
end





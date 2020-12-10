#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
Note: All tests should be run from a julia REPR within the julia folder, using:
      julia --project=.
      include("test/all.jl")
      To see coverage, run with:
      julia --project=. --code-coverage=tracefile-%p.info --code-coverage=user
      include("test/all.jl")
=#

include("load.jl");

@testset "OptimalConstraintTree" begin
    include(string(PROJECT_ROOT, "/src.jl"))

#     include(string(PROJECT_ROOT, "/cbf.jl"))

    include(string(PROJECT_ROOT, "/sagemark.jl"))

#     include(string(PROJECT_ROOT, "/speedreducer.jl"))

#     include(string(PROJECT_ROOT, "/lse.jl"))

    include(string(PROJECT_ROOT, "/benchmarks/gams.jl"))
end

# Other tests to try later

# include("test/test_transonic.jl");
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
#     include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/src.jl"))

#     include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/cbf.jl"))

#     include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/sagemark.jl"))

#     include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/speedreducer.jl"))

#     include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/lse.jl"))

    include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/gams.jl"))
end

# Other tests to try later

# include("/test/test_transonic.jl");
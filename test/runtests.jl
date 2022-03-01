#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
Note: All tests should be run from a julia REPR within the OCTHaGOn folder, using:
      julia --project=.
      include("test/all.jl")
      
      To see coverage, run with:
      julia --project=. --code-coverage=tracefile-%p.info --code-coverage=user
      include("test/all.jl")
=#

using OCTHaGOn

N = 5
M = 1

m = JuMP.Model(with_optimizer(CPLEX_SILENT))
@variable(m, x[1:N+1])
@objective(m, Min, x[N+1])

# Initialize GlobalModel
gm = OCTHaGOn.GlobalModel(model = m, name = "qp")

Q = randn(N, N)
c = randn(N)
expr = :((x) -> -x'*$(Q)*x - $(c)'*x)

lbs = push!(-5*ones(N), -800)
ubs = push!(5*ones(N), 0)
lbs_dict = Dict(x .=> lbs)
ubs_dict = Dict(x .=> ubs)

OCTHaGOn.bound!(gm, Dict(var => [lbs_dict[var], ubs_dict[var]] for var in gm.vars))
        

# Add constraints
#OCTHaGOn.add_nonlinear_constraint(gm, expr, vars=x[1:N])
OCTHaGOn.add_nonlinear_constraint(gm, expr, vars=x[1:N], expr_vars=[x[1:N]])

OCTHaGOn.globalsolve!(gm)


# include("load.jl");

# @testset "OCTHaGOn" begin
#     include(string(OCTHaGOn.PROJECT_ROOT, "/test/src.jl"))

#     include(string(OCTHaGOn.PROJECT_ROOT, "/test/imports.jl"))

# #     include(string(OCTHaGOn.PROJECT_ROOT, "/test/cbf.jl"))

#     include(string(OCTHaGOn.PROJECT_ROOT, "/test/algorithms.jl"))

# #     include(string(OCTHaGOn.PROJECT_ROOT, "/test/lse.jl"))

# end

# Other tests to try later

# include("/test/test_transonic.jl");
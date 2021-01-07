#=
load:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-09-04
=#

using DataFrames
using JuMP
using BARON
using CPLEX
# using Gurobi
# using Ipopt
using MathOptInterface
using Test
using Random

include("../src/OptimalConstraintTree.jl")
using .OptimalConstraintTree
global OCT = OptimalConstraintTree
global MOI = MathOptInterface
global CPLEX_SILENT = with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0)
global BARON_SILENT = with_optimizer(BARON.Optimizer, OutputFlag = 0)
# global IPOPT_SILENT = with_optimizer(Ipopt.Optimizer, print_level = 0)
# global GUROBI_SILENT = with_optimizer(Gurobi.Optimizer, OutputFlag = 0, Gurobi.Env())
Random.seed!(1);
MOI.Silent() = true;

include(OCT.PROJECT_ROOT * "/test/tools/gams.jl");

include(OCT.PROJECT_ROOT * "/test/tools/sagemark.jl");

include(OCT.PROJECT_ROOT * "/test/tools/models.jl")

include(OCT.PROJECT_ROOT * "/data/baron/gear.jl");

include(OCT.PROJECT_ROOT * "/data/baron/minlp.jl");
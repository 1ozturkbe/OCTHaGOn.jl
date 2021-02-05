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
using JLD
using MathOptInterface
using Test
using Random

include("../src/OptimalConstraintTree.jl")
using .OptimalConstraintTree
global OCT = OptimalConstraintTree
global MOI = MathOptInterface
global BARON_SILENT = with_optimizer(BARON.Optimizer, OutputFlag = 0)
global CPLEX_SILENT = OCT.CPLEX_SILENT
# global GUROBI_SILENT = with_optimizer(Gurobi.Optimizer, OutputFlag = 0, Gurobi.Env())
Random.seed!(1);
MOI.Silent() = true;

include(OCT.PROJECT_ROOT * "/test/tools/gams.jl");

include(OCT.PROJECT_ROOT * "/test/tools/sagemark.jl");

include(OCT.PROJECT_ROOT * "/test/tools/models.jl")

include(OCT.DATA_DIR * "speed_reducer.jl")

include(OCT.BARON_DIR * "gear.jl");

include(OCT.BARON_DIR * "minlp.jl");

include(OCT.BARON_DIR * "nlp1.jl");

include(OCT.BARON_DIR * "nlp2.jl");

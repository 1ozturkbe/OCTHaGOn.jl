#=
load:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-09-04
=#

using DataFrames
using Gurobi
using Ipopt
using JuMP
using MathOptInterface
using Test
using Random

include("../src/OptimalConstraintTree.jl")
using .OptimalConstraintTree
global OCT = OptimalConstraintTree
global MOI = MathOptInterface
global IPOPT_SILENT = with_optimizer(Ipopt.Optimizer, print_level = 0)
global GUROBI_SILENT = with_optimizer(Gurobi.Optimizer, OutputFlag = 0, Gurobi.Env())
Random.seed!(1);
MOI.Silent() = true;

include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/tools/gams.jl"));

include(string(OptimalConstraintTree.PROJECT_ROOT, "/test/tools/sagemark.jl"));
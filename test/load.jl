#=
load:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-04
=#

using DataFrames
using Gurobi
using JuMP
using MathOptInterface
using Test
using Random

include("../src/OptimalConstraintTree.jl")
using .OptimalConstraintTree
global OCT = OptimalConstraintTree
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__
Random.seed!(1);
MOI.Silent() = true
#=
load:
- Julia version: 1.5.1
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
Random.seed!(1);
MOI.Silent() = true
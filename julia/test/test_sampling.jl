#=
test_sampling:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-07
=#

using Test
using Combinatorics
using Gurobi
using JuMP
using LatinHypercubeSampling
using MosekTools
using MathOptInterface
using NearestNeighbors

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__

# Import a model
filename = string("../data/cblib.zib.de/shortfall_20_15.cbf.gz");
md = OCT.CBF_to_ModelData(filename);
OCT.find_bounds!(md)
OCT.update_bounds!(md, lbs = Dict(vk => 0 for vk in md.vks));
OCT.update_bounds!(md, ubs = Dict(vk => 1 for vk in md.vks));

# Pick one of the functions, and sample.
bbf = md.fns[1];
bbf.n_samples= 500;
OCT.sample_and_eval!(bbf); # initial is uniform!
println("Feasibility: ", bbf.feas_ratio)
OCT.sample_and_eval!(bbf); # then knn!
println("Feasibility: ", bbf.feas_ratio)
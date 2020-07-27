#=
test_bbf:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
Tests BlackBoxFn and its functions
=#

include("../src/OptimalConstraintTree.jl")
using Test
using GaussianProcesses

md = OCT.sagemark_to_ModelData(3, lse=false);
md.lbs[end] = -300;
md.ubs[end]= -0;

n_samples = 20;
X = OCT.sample(md, n_samples = n_samples);

bbf = md.fns[1];

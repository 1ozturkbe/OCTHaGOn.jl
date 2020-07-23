#=
test_gp:
- Julia version: 
- Author: Berk
- Date: 2020-07-22
Testing background Gaussian Processes (GP)
=#

using Test
using Gurobi
using MosekTools
using JuMP
using MathOptInterface
using GaussianProcesses
using Plots

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__

# Try fitting a GP on a simple example
md = OCT.sagemark_to_ModelData(3, lse=false);
md.lbs[end] = -300;
md.ubs[end]= -0;
n_samples = 200;
X = OCT.sample(md, n_samples = n_samples);
Y = [md.ineq_fns[1](X[j,:]) for j = 1:n_samples];

# Set-up mean and kernel, and then the GP
se = SE(1., 1.)
m = MeanZero()
gp = GP(X', Y, m, se)
# Plot mean and variance
# p1 = plot(gp; title="Mean of GP")
# p2 = plot(gp; var=true, title="Variance of GP", fill=true)
# plot(p1, p2; fmt=:png)
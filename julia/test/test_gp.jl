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
n_samples = 20;
X = OCT.sample(md, n_samples = n_samples);
fn = md.ineq_fns[1];
Y = [fn(X[j,:]) for j = 1:n_samples];

# Set-up mean and kernel, and then the GP
se = SE(1., 1.)
m = MeanZero()
gp = GP(X', Y, m, se)
dense_X = OCT.sample(md, n_samples=1000);

# Optimize GP, and then predict dense_X locations
optimize!(gp);
μ_p, k_p = predict_f(gp, Matrix(dense_X'));

# Functions that might be useful from the documentation:
# http://stor-i.github.io/GaussianProcesses.jl/latest/
# mcmc (Markov Chain Monte Carlo)
# optimize (for the optimization of parameters of the GP)

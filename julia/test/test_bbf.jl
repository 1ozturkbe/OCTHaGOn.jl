#=
test_bbf:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
Tests BlackBoxFn and its functions, including GaussianProcesses
=#
using Test
using GaussianProcesses

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__


md = OCT.sagemark_to_ModelData(3, lse=false);
md.lbs[end] = -300;
md.ubs[end]= -0;

n_samples = 20;
X = OCT.sample(md, n_samples = n_samples);

bbf = md.fns[1];
Y = [bbf(X[j,:]) for j = 1:n_samples];

# Set-up mean and kernel, and then the GP
X = OCT.sample(md, n_samples=20);
OCT.eval!(bbf, X);
OCT.update_bounds!(bbf, md.lbs, md.ubs)

# Optimize GP, and then predict dense_X locations
OCT.optimize_gp!(bbf)
dense_X = OCT.sample(md, n_samples = 1000);
μ_p, k_p = OCT.predict(bbf, dense_X);


# Functions that might be useful from the documentation:
# http://stor-i.github.io/GaussianProcesses.jl/latest/
# mcmc (Markov Chain Monte Carlo)
# optimize (for the optimization of parameters of the GP)

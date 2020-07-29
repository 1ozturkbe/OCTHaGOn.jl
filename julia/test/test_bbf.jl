#=
test_bbf:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
Tests BlackBoxFn and its functions, including GaussianProcesses
=#
using Test
using GaussianProcesses
using Plots

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree;
global PROJECT_ROOT = @__DIR__

bbf = OCT.BlackBoxFn(fn = x -> x[1]^2 * sin(x[1]) + 2,
                    idxs = [1], lbs = [-5.], ubs = [5.]);

# Sampling and plotting raw data.
n_samples = 5;
OCT.sample_and_eval!(bbf, n_samples=n_samples);
OCT.plot(bbf)

# Optimize first gp
OCT.optimize_gp!(bbf)
plot(bbf.gp)
#
# # Sample and plot again
OCT.sample_and_eval!(bbf, n_samples=n_samples);
OCT.optimize_gp!(bbf)
plot(bbf.gp)



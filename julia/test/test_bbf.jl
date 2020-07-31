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
                    idxs = [1], lbs = [-5.], ubs = [5.],
                    n_samples = 5);
x = range(-5,stop=5, step=0.1);
plot(x, bbf.fn.(x))

# Sampling and plotting raw data.
OCT.sample_and_eval!(bbf);
OCT.optimize_gp!(bbf)
plot(bbf.gp)

# # Sample and plot again
@time OCT.sample_and_eval!(bbf);
@time OCT.optimize_gp!(bbf)
plot(x, bbf.fn.(x))
plot!(bbf.gp)



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
using Random

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree;
global PROJECT_ROOT = @__DIR__

bbf = OCT.BlackBoxFn(fn = x -> x[:x1] + x[:x3]^3,
                    vks = [:x1, :x3], lbs = Dict(:x1 => -5, :x3 => -5),
                                      ubs = Dict(:x1 => 5, :x3 => 5),
                    n_samples = 50);

# Check evaluation of samples from Dict and DataFrame
samples = DataFrame(rand(4,4), [Symbol("x",i) for i=1:4]);
val1 = bbf(samples[1,:]);
sample = Dict(:x1 => samples[1,1], :x3 => samples[1,3])
val2 = bbf(sample);
@test val1 == val2 == samples[1,1] + samples[1,3]^3;

# Check evaluation
OCT.sample_and_eval!(bbf)

# bbf = OCT.BlackBoxFn(fn = x -> x[1]^2 * sin(x[1]) + 2,
#                     idxs = [1], lbs = [-5.], ubs = [5.],
#                     n_samples = 50);
# x = range(-5,stop=5, step=0.1);
# plot(x, bbf.fn.(x))

# Sampling and plotting raw data.
# OCT.sample_and_eval!(bbf);
# OCT.optimize_gp!(bbf)
# plot!(bbf.gp)
#
# # # Sample and plot again
# OCT.sample_and_eval!(bbf);
# OCT.optimize_gp!(bbf)
# plot!(bbf.gp)
#
# # Finally learning constraint
# OCT.learn_constraint!(bbf);
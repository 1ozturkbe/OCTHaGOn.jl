#=
test_bbf:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
Tests BlackBoxFn and its functions, including GaussianProcesses
=#
using DataFrames
using Test
using GaussianProcesses
using Plots
using Random

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree;
global PROJECT_ROOT = @__DIR__

bbf = OCT.BlackBoxFn(fn = x -> x[:x1] + x[:x3]^3,
                    vks = [:x1, :x3], n_samples = 50);

# Check evaluation of samples from Dict, DataFrameRow and DataFrame
samples = DataFrame(rand(4,4), [Symbol("x",i) for i=1:4]);
val1 = bbf(samples[1,:]);
sample = Dict(:x1 => samples[1,1], :x3 => samples[1,3])
val2 = bbf(sample);
val3 = bbf(samples);
@test val1 == val2 == val3[1] == samples[1,1] + samples[1,3]^3;

# Check unbounded sampling
@test_throws OCT.OCTException OCT.sample_and_eval!(bbf);
#
# # Check proper bounding
lbs = Dict(:x1 => -5, :x3 => -5);
ubs = Dict(:x1 => 5, :x3 => 5);
OCT.update_bounds!(bbf, lbs=lbs, ubs=ubs);
@test bbf.ubs == ubs
lbs = Dict(:x1 => -3, :x2 => 2); # check update with vk not in bbf
OCT.update_bounds!(bbf, lbs=lbs);
@test !(:x2 in keys(md.lbs))
@test bbf.lbs[:x1] == -3;
@test_throws OCT.OCTException OCT.update_bounds!(bbf, ubs = Dict(:x1 => -6)) # check infeasible bounds
@test bbf.ubs[:x1] == 5;
#
#
#
# # Check sampling and plotting in 1D
# bbf = OCT.BlackBoxFn(fn = x -> x[:x1]^2 * sin(x[:x1]) + 2,
#                      vks = [:x1], lbs = Dict(:x1 => -5), ubs = Dict(:x1 => 5),
#                     n_samples = 20);
# x = DataFrame(:x1 => range(-5,stop=5, step=0.1));
# plot(Array(x), bbf(x));

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
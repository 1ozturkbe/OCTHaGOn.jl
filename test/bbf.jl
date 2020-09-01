#=
test_bbf:
- Julia version: 
- Author: Berk
- Date: 2020-07-27
Tests BlackBoxFunction and its functions, including GaussianProcesses
=#


bbf = BlackBoxFunction(fn = x -> x[:x1] + x[:x3]^3,
                    vks = [:x1, :x3], n_samples = 50);

# Check evaluation of samples from Dict, DataFrameRow and DataFrame
samples = DataFrame(rand(4,4), [Symbol("x",i) for i=1:4]);
val1 = bbf(samples[1,:]);
sample = Dict(:x1 => samples[1,1], :x3 => samples[1,3])
val2 = bbf(sample);
val3 = bbf(samples);
sample = Dict(:x1 => samples[1,1], :x2 => 2, :x3 => samples[1,3]);
val4 = bbf(sample);
sample = DataFrame(sample);
val5 = bbf(sample);
@test val1 == val2 == val3[1] == val4 == val5[1] == samples[1,1] + samples[1,3]^3;

# Check unbounded sampling
@test_throws OCTException sample_and_eval!(bbf);

# # Check proper bounding
lbs = Dict(:x1 => -5, :x3 => -5);
ubs = Dict(:x1 => 5, :x3 => 5);
update_bounds!(bbf, lbs=lbs, ubs=ubs);
@test bbf.ubs == ubs
lbs = Dict(:x1 => -3, :x2 => 2); # check update with vk not in bbf
update_bounds!(bbf, lbs=lbs);
@test !(:x2 in keys(bbf.lbs))
@test bbf.lbs[:x1] == -3;
@test_throws OCTException update_bounds!(bbf, ubs = Dict(:x1 => -6)) # check infeasible bounds
@test bbf.ubs[:x1] == 5;

# Check sampling and plotting in 1D or 2D
# bbf = BlackBoxFunction(fn = x -> x[:x1]^2 * sin(x[:x1]) + 2,
#                      vks = [:x1], lbs = Dict(:x1 => -5), ubs = Dict(:x1 => 5),
#                     n_samples = 20);

bbf = BlackBoxFunction(fn = x -> x[:x1]^2 + x[:x1]*sin(x[:x1]* x[:x2]) -5,
 vks = [:x1, :x2], lbs = Dict([:x1, :x2] .=> -5), ubs = Dict([:x1, :x2] .=> 5),
n_samples = 100);
sample_and_eval!(bbf);

# Sampling and plotting raw data.
sample_and_eval!(bbf);
# plot_2d(bbf);

# Learning constraint
learn_constraint!(bbf);
IAI.show_in_browser(bbf.learners[end].lnr);

# Showing correct vs incorrect predictions
# plot_2d_predictions(bbf);


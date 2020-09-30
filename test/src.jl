#=
test_src:
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code,
without any machine learning components
All the rest of the tests look at examples
=#

##########################
# BLACKBOXFUNCTION TESTS #
##########################

model = Model()
@variables(model, begin
    -5 <= x[1:5] <= 5
    -4 <= y[1:3] <= 1
    -30 <= z
end)
ex = @NLexpression(model, sum(x[i] for i=1:4) - y[1] * y[2] + z)
@NLconstraint(model, ex <= 10)
@constraint(model, sum(x[4]^2 + x[5]^2) <= z)
@constraint(model, sum(y[:]) >= -2)

# Testing getting variables
varkeys = ["x[1]", x[1], :z, :x];
vars = [x[1], x[1], z, x[:]];
@test all(vars .==  fetch_variable.(model, varkeys))

# Testing bounds and fixing variables
bounds = get_bounds(model);
@test bounds[z] == [-30., Inf]
bounds = Dict(:x => [-10,1], z => [-10, 10])
bound!(model, bounds)
@test get_bounds(model)[z] == [-10, 10]
@test get_bounds(model)[x[3]] == [-5, 1]

# Testing ways of "sanitizing data" and also evaluating
inp = Dict(x[1] => 1, x[2] => 2, x[3] => 3, x[4] => 4, y[1] => 5, y[2] => 6, z => 7)
inp_dict = Dict(string(key) => value for (key, value) in inp)
inp_df = DataFrame(inp_dict)
@test sanitize_data(model, inp) == sanitize_data(model, inp_dict) == sanitize_data(model, inp_df) == inp_df

# Testing separation of constraints
con

@test evaluate(model, inp) == evaluate(model, inp_dict) == evaluate(model, inp_df) == -13.0

# Testing variable processing
vars = JuMP.all_variables(model);
vks = string.(vars);

# Ideally,  would like to be able to do:
inp = Dict(subset_of_vars .=> nums)
evaluate(constraint, dict)

# m = Model(Ipopt.Optimizer)
# @variable(m, x[1:2])
# # Use Meta.parse and custom transformations to create this object from a string. eval() is not needed.
# expr = :($(x[1])^3 + 5.0 * $(x[1]) * $(x[2]))
# add_NL_constraint(m, :($expr <= 5))
# https://discourse.julialang.org/t/procedural-nonlinear-constaint-generation-string-to-nlconstraint/34799/3


# Check evaluation of samples from Dict, DataFrameRow and DataFrame
samples = DataFrame(rand(4,4), string.(x[1:4]))
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
lbs = Dict(:x1 => -3, :x2 => 2); # check update with vk not in BlackBoxFn
update_bounds!(bbf, lbs=lbs);
@test !(:x2 in keys(bbf.lbs))
@test bbf.lbs[:x1] == -3;
@test_throws OCTException update_bounds!(bbf, ubs = Dict(:x1 => -6)) # check infeasible bounds
@test bbf.ubs[:x1] == 5;

# Check sampling and plotting in 1D or 2D (plotting disabled for faster testing)
bbf = BlackBoxFunction(fn = x -> x[:x1]^2 + x[:x1]*sin(x[:x1]* x[:x2]) -5,
 vks = [:x1, :x2], lbs = Dict([:x1, :x2] .=> -5), ubs = Dict([:x1, :x2] .=> 5),
n_samples = 100);
sample_and_eval!(bbf);

# Sampling, learning and showing...
# plot_2d(bbf);
learn_constraint!(bbf);
# show_trees(bbf);

# Showing correct vs incorrect predictions
# plot_2d_predictions(bbf);

###################
# MODELDATA TESTS #
###################

# Initialization tests
md = ModelData(c = [1,2,3]);

# Check infeasible bounds for Model Data
lbs = Dict(md.vks .=> [-2,1,3]);
update_bounds!(md, lbs=lbs);
# Check sampling unbounded model exception
@test_throws OCTException X = lh_sample(md, n_samples=100);
ubs = Dict(md.vks .=> [4, 5, 6]);
update_bounds!(md, ubs=ubs);

# Check JuMP variable and bound creation
@test all([md.lbs[vk] == lower_bound(md.vars[vk]) for vk in md.vks])
@test all([md.ubs[vk] == upper_bound(md.vars[vk]) for vk in md.vks])

# Check invalid bounds exception
ubs = Dict(md.vks .=> [Inf, -1, 6]);
@test_throws OCTException update_bounds!(md, ubs = ubs);

# Check sampling Model Data
lh_sample(md, n_samples=50);

# Check tautological and infeasible constraints
add_fn!(md, BlackBoxFunction(fn = x -> 1,
                    vks = [:x1, :x3], n_samples = 50));
add_fn!(md, BlackBoxFunction(fn = x -> -1,
                    vks = [:x1, :x2], n_samples = 50));
sample_and_eval!(md)
@test feasibility(md) == [1., 0.];
@test accuracy(md) == [1., 1.];
learn_constraint!(md);
globalsolve(md);
@test all([solution(md)[vk][1] == md.lbs[vk] for vk in md.vks])

# TODO: add check for number of linear constraints in md.JuMP_Model.

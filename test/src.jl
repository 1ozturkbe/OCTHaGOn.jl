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

# Bounds and fixing variables
bounds = get_bounds(model);
@test bounds[z] == [-30., Inf]
bounds = Dict(:x => [-10,1], z => [-10, 10])
bound!(model, bounds)
@test get_bounds(model)[z] == [-10, 10]
@test get_bounds(model)[x[3]] == [-5, 1]

# Linearization of objective
linearize_objective!(model);
@objective(model, Min, x[3]^2)
linearize_objective!(model);
@test JuMP.objective_function(model) isa JuMP.VariableRef

# "sanitizing data"
inp = Dict(x[1] => 1., x[2] => 2, x[3] => 3, x[4] => 4, x[5] => 1, y[1] => 5, y[2] => -6, y[3] => -7, z => 7)
inp_dict = Dict(string(key) => value for (key, value) in inp)
inp_df = DataFrame(inp_dict)
@test sanitize_data(model, inp) == sanitize_data(model, inp_dict) == sanitize_data(model, inp_df) == inp_df

# Separation of constraints
l_constrs, nl_constrs = classify_constraints(model)
@test length(l_constrs) == 20 && length(nl_constrs) == 2

# Set constants
sets = [MOI.GreaterThan(2), MOI.EqualTo(0), MOI.SecondOrderCone(3), MOI.GeometricMeanCone(2), MOI.SOS1([1,2,3])]
@test get_constant.(sets) == [2, 0, nothing, nothing, nothing]

# Evaluation
@test evaluate(l_constrs[1], inp) == evaluate(l_constrs[1], inp_dict) == evaluate(l_constrs[1], inp_df) == -6.
@test evaluate(nl_constrs[1], inp) == evaluate(nl_constrs[1], inp_dict) == evaluate(nl_constrs[1], inp_df) == -10.
inp_df = DataFrame(-5 .+ 10 .*rand(3, size(inp_df,2)), string.(keys(inp)))
@test evaluate(l_constrs[1], inp_df) == inp_df["y[1]"] + inp_df["y[2]"] + inp_df["y[3]"] .+ 2.

# BBF creation
bbf = BlackBoxFunction(constraint = nl_constrs[1], vars = [x[4], x[5], z])

# Check evaluation of samples
samples = DataFrame(randn(10, length(bbf.vars)),string.(bbf.vars))
vals = bbf(samples);
@test vals ≈ -1*samples["x[4]"].^2 - samples["x[5]"].^2 + samples["z"]

# Checks different kinds of sampling
X_bound = boundary_sample(bbf);
@test size(X_bound, 1) == 2^(length(bbf.vars)+1)
@test_throws OCTException knn_sample(bbf, k=3)
X_lh = lh_sample(bbf);

# Check sample_and_eval
sample_and_eval!(bbf);
sample_and_eval!(bbf);

# Sampling, learning and showing...
# plot_2d(bbf);
learn_constraint!(bbf);
# show_trees(bbf);

# Showing correct vs incorrect predictions
# plot_2d_predictions(bbf);

# Check infeasible bounds
new_bounds = Dict(x[4] => [-10,-6])
@test_throws OCTException OptimalConstraintTree.check_infeasible_bounds(model, new_bounds)
@test_throws OCTException bound!(model, new_bounds)

# Check unbounded sampling
JuMP.delete_lower_bound(z)
@test_throws OCTException X = lh_sample(bbf, n_samples=100);
bound!(model, Dict(z => [-10,10]))

# Check feasibility and accuracy
@test 0 <= feasibility(bbf) <= 1
@test 0 <= accuracy(bbf) <= 1
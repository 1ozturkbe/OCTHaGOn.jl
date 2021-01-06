#=
test_src:
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code,
without *many* machine learning components
=#

##########################
# SOURCE TESTS #
##########################

""" Tests expression parsing. """
function test_expressions()
    model, x, y, z, a = test_model()
    expr = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)
    simp_expr = :(x -> sum(5 .* x))
    f = functionify(expr)

    # Testing function evaluation
    res = Base.invokelatest(f, (ones(4), ones(2),5)...)
    @test res isa Float64
    res = Base.invokelatest(f, (x, y, z)...)
    @test res isa JuMP.GenericQuadExpr
    @test res == sum(x[1:4]) - y[1] * y[2] + z

    # Testing variable input parsing from expression
    @test vars_from_expr(expr, model) == [x,y,z]
    @test vars_from_expr(simp_expr, model) == [x]
    @test vars_from_expr(:((x, a) -> x[2] + a[2,2]),model) == [x, a]

    # Testing "flattening of expressions" for nonlinearization
    vars = vars_from_expr(expr, model)
    var_ranges = [(1:5),(6:8),9]
    flat_expr = :((x...) -> $(expr)([x[i] for i in $(var_ranges)]...))
    fn = functionify(flat_expr)
    @test Base.invokelatest(fn, [1,2,3,4,1,5,-6,-7,7]...) == Base.invokelatest(f, ([1,2,3,4,1], [5,-6,-7], 7)...)
    @test Base.invokelatest(fn, flat(vars)...) == res

    # Testing proper mapping for expressions
    flatvars = flat([y[2], z, x[1:4]])
    vars = vars_from_expr(expr, model)
    @test get_varmap(vars, flatvars) == [(2,2), (3,0), (1,1), (1, 2), (1,3), (1,4)]
    @test infarray([(1,4), (1,3), (2,0)]) == [[Inf, Inf, Inf, Inf], Inf]
end

function test_variables()
    model, x, y, z, a  = test_model()

    # Testing getting variables
    varkeys = ["x[1]", x[1], :z, :x];
    vars = [x[1], x[1], z, x[:]];
    @test all(vars .==  fetch_variable(model, varkeys))
end

function test_bounds()
    model, x, y, z, a = test_model()

    # Bounds and fixing variables
    bounds = get_bounds(model);
    @test bounds[z] == [-30., Inf]
    bounds = Dict(:x => [-10,1], z => [-10, 10])
    bound!(model, bounds)
    @test get_bounds(model)[z] == [-10, 10]
    @test get_bounds(model)[x[3]] == [-5, 1]

    # Check infeasible bounds
    new_bound = x[4] => [-10,-6]
    @test_throws OCTException OCT.check_infeasible_bound(new_bound)
    @test_throws OCTException bound!(model, new_bound)

    # Check unbounds
    @test all(collect(values(get_unbounds(model)))[i] == [-2., Inf] for i = 1:4)

    # Check that JuMP does not set Inf bounds
    m = JuMP.Model()
    @variable(m, x[1:2])
    @variable(m, y >= 5)
    bound!(m, Dict(var => [-Inf, Inf] for var in all_variables(m)))
    @test !JuMP.has_lower_bound(x[1]) && JuMP.has_lower_bound(y) && !JuMP.has_upper_bound(x[2])
    bound!(m, Dict(var => [-10, 10] for var in all_variables(m)))
    @test isnothing(get_unbounds(m))
end

function test_sets()
    sets = [MOI.GreaterThan(2), MOI.EqualTo(0), MOI.SecondOrderCone(3), MOI.GeometricMeanCone(2), MOI.SOS1([1,2,3])]
    @test get_constant.(sets) == [2, 0, nothing, nothing, nothing]
end

function test_linearize()
    model, x, y, z, a = test_model()
    # Linearization of objective
    linearize_objective!(model);
    @objective(model, Min, x[3]^2)
    linearize_objective!(model);
    @test JuMP.objective_function(model) isa JuMP.VariableRef

    @constraint(model, sum(x[4]^2 + x[5]^2) <= z)
    @constraint(model, sum(y[:]) >= -2)
    # Separation of model constraints
    l_constrs, nl_constrs = classify_constraints(model)
    @test length(l_constrs) == 27 && length(nl_constrs) == 1
end

function test_bbf()
    model, x, y, z, a = test_model()
    nl_constr = @constraint(model, sum(x[4]^2 + x[5]^2) <= z)
    expr = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)

    # "Sanitizing" data
    inp = Dict(x[1] => 1., x[2] => 2, x[3] => 3, x[4] => 4, x[5] => 1, y[1] => 5, y[2] => -6, y[3] => -7, z => 7)
    inp_dict = Dict(string(key) => val for (key, val) in inp)
    inp_df = DataFrame(inp_dict)
    @test data_to_DataFrame(inp) == data_to_DataFrame(inp_dict) == data_to_DataFrame(inp_df) == inp_df
    @test data_to_Dict(inp_df, model) == data_to_Dict(inp, model) == data_to_Dict(inp_dict, model) == inp

    # Test BBF creation
    @test isnothing(functionify(nl_constr))
    @test functionify(expr) isa Function
    bbfs = [BlackBoxClassifier(constraint = nl_constr, vars = [x[4], x[5], z]),
        BlackBoxClassifier(constraint = expr, vars = flat([x[1:4], y[1:2], z]),
                         expr_vars = [x,y,z])]

    # Evaluation (scalar)
    # Quadratic (JuMP compatible) constraint
    @test evaluate(bbfs[1], inp) == evaluate(bbfs[1], inp_dict) == evaluate(bbfs[1], inp_df) == -10.
    # Nonlinear expression
    @test evaluate(bbfs[2], inp) == evaluate(bbfs[2], inp_dict) == evaluate(bbfs[2], inp_df)

    # Evaluation (vector)
    inp_df = DataFrame(-5 .+ 10 .*rand(3, size(inp_df,2)), string.(keys(inp)))
    inp_dict = data_to_Dict(inp_df, model)
    @test evaluate(bbfs[1], inp_dict) == evaluate(bbfs[1], inp_df) == inp_df[!, "z"] -
                                                    inp_df[!, "x[4]"].^2 - inp_df[!, "x[5]"].^2
    @test evaluate(bbfs[2], inp_dict) == evaluate(bbfs[2], inp_df)

    # BBF CHECKS
    bbf = bbfs[1]

    # Check unbounded sampling
    @test_throws OCTException X = lh_sample(bbf, n_samples=100);

    # Check knn_sampling without previous samples
    @test_throws OCTException knn_sample(bbf, k=3)

    # Check evaluation of samples
    samples = DataFrame(randn(10, length(bbf.vars)),string.(bbf.vars))
    vals = bbf(samples);
    @test vals ≈ -1*samples[!, "x[4]"].^2 - samples[!, "x[5]"].^2 + samples[!, "z"]

    # Checks different kinds of sampling
    bound!(model, Dict(z => [-Inf, 10]))
    X_bound = boundary_sample(bbf);
    @test size(X_bound, 1) == 2^(length(bbf.vars)+1)
    @test_throws OCTException knn_sample(bbf, k=3)
    X_lh = lh_sample(bbf, lh_iterations=3);

    # Check sample_and_eval
    uniform_sample_and_eval!(bbf);

    # Sampling, learning and showing...
    learn_constraint!(bbf);

    # Check feasibility and accuracy
    @test 0 <= feasibility(bbf) <= 1
    @test 0 <= accuracy(bbf) <= 1

    # Training a model
    mi_constraints, leaf_variables = add_feas_constraints!(model, bbf.vars, bbf.learners[1].lnr);
    @test true
end

""" Testing some IAI kwarging. """
function test_kwargs()
    # Classification kwargs first...
    sample_kwargs = Dict(:validation_criterion => :sensitivity, 
                       :localsearch => false, 
                       :invalid_kwarg => :hello,
                       :ls_num_tree_restarts => 20)

    dict_fit = fit_kwargs(; sample_kwargs...)
    dict_fit2 = fit_kwargs(validation_criterion = :sensitivity, localsearch = false, invalid_kwarg = :hello,
                           ls_num_tree_restarts = 20)
    @test dict_fit == dict_fit2 == Dict(:validation_criterion => :sensitivity, :positive_label => 1)

    dict_lnr = lnr_kwargs(; sample_kwargs...)
    dict_lnr2 = lnr_kwargs(validation_criterion = :sensitivity, localsearch = false, invalid_kwarg = :hello,
                           ls_num_tree_restarts = 20)
    @test dict_lnr == dict_lnr2 == Dict(:localsearch => false, :ls_num_tree_restarts => 20)

    # Regression kwargs next...
    dict_fit = fit_kwargs(true; sample_kwargs)
    @test dict_fit == Dict(:validation_criterion => :mse)
end

test_expressions()

test_variables()

test_bounds()

test_sets()

test_linearize()

test_bbf()

test_kwargs()
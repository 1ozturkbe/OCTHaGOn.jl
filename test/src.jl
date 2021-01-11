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

function test_bbl()
    model, x, y, z, a = test_model()
    nl_constr = @constraint(model, sum(x[4]^2 + x[5]^2) <= z)
    expr = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)

    # "Sanitizing" data
    inp = Dict(x[1] => 1., x[2] => 2, x[3] => 3, x[4] => 4, x[5] => 1, y[1] => 5, y[2] => -6, y[3] => -7, z => 7)
    inp_dict = Dict(string(key) => val for (key, val) in inp)
    inp_df = DataFrame(inp_dict)
    @test data_to_DataFrame(inp) == data_to_DataFrame(inp_dict) == data_to_DataFrame(inp_df) == inp_df
    @test data_to_Dict(inp_df, model) == data_to_Dict(inp, model) == data_to_Dict(inp_dict, model) == inp

    # Test bbl creation
    @test isnothing(functionify(nl_constr))
    @test functionify(expr) isa Function
    bbls = [BlackBoxClassifier(constraint = nl_constr, vars = [x[4], x[5], z]),
        BlackBoxClassifier(constraint = expr, vars = flat([x[1:4], y[1:2], z]),
                         expr_vars = [x,y,z])]

    # Evaluation (scalar)
    # Quadratic (JuMP compatible) constraint
    @test evaluate(bbls[1], inp) == evaluate(bbls[1], inp_dict) == evaluate(bbls[1], inp_df) == -10.
    # Nonlinear expression
    @test evaluate(bbls[2], inp) == evaluate(bbls[2], inp_dict) == evaluate(bbls[2], inp_df)

    # Evaluation (vector)
    inp_df = DataFrame(-5 .+ 10 .*rand(3, size(inp_df,2)), string.(keys(inp)))
    inp_dict = data_to_Dict(inp_df, model)
    @test evaluate(bbls[1], inp_dict) == evaluate(bbls[1], inp_df) == inp_df[!, "z"] -
                                                    inp_df[!, "x[4]"].^2 - inp_df[!, "x[5]"].^2
    @test evaluate(bbls[2], inp_dict) == evaluate(bbls[2], inp_df)

    # bbl CHECKS
    bbl = bbls[1]

    # Check unbounded sampling
    @test_throws OCTException X = lh_sample(bbl, n_samples=100);

    # Check knn_sampling without previous samples
    @test_throws OCTException knn_sample(bbl, k=3)

    # Check evaluation of samples
    samples = DataFrame(randn(10, length(bbl.vars)),string.(bbl.vars))
    vals = bbl(samples);
    @test vals ≈ -1*samples[!, "x[4]"].^2 - samples[!, "x[5]"].^2 + samples[!, "z"]

    # Checks different kinds of sampling
    bound!(model, Dict(z => [-Inf, 10]))
    X_bound = boundary_sample(bbl);
    @test size(X_bound, 1) == 2^(length(bbl.vars)+1)
    @test_throws OCTException knn_sample(bbl, k=3)
    X_lh = lh_sample(bbl, lh_iterations=3);

    # Check sample_and_eval
    uniform_sample_and_eval!(bbl);

    # Sampling, learning and showing...
    learn_constraint!(bbl);

    # Check feasibility and accuracy
    @test 0 <= feasibility(bbl) <= 1
    @test 0 <= accuracy(bbl) <= 1

    # Training a model
    mi_constraints, leaf_variables = add_feas_constraints!(model, bbl.vars, bbl.learners[1]);
    @test true
end

""" Testing some IAI kwarging. """
function test_kwargs()
    # Classification kwargs first...
    sample_kwargs = Dict(:localsearch => false, 
                       :invalid_kwarg => :hello,
                       :ls_num_tree_restarts => 20)

    dict_fit = fit_classifier_kwargs(; sample_kwargs...)
    dict_fit2 = fit_classifier_kwargs(localsearch = false, invalid_kwarg = :hello,
                           ls_num_tree_restarts = 20)
    @test dict_fit == dict_fit2 == Dict(:sample_weight => :autobalance)

    dict_lnr = classifier_kwargs(; sample_kwargs...)
    dict_lnr2 = classifier_kwargs(localsearch = false, invalid_kwarg = :hello, ls_num_tree_restarts = 20)
    @test dict_lnr == dict_lnr2 == Dict(:localsearch => false, :ls_num_tree_restarts => 20)

    # Regression kwargs next...
    dict_fit = fit_regressor_kwargs(; sample_kwargs)
    @test dict_fit == Dict()

    dict_lnr = regressor_kwargs(; sample_kwargs...)
    dict_lnr2 = regressor_kwargs(localsearch = false, invalid_kwarg = :hello, ls_num_tree_restarts = 20)
    @test dict_lnr == dict_lnr2 == Dict(:localsearch => false, :ls_num_tree_restarts => 20)
end

""" Tests basic functionalities in GMs. """
function test_basic_gm()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    uniform_sample_and_eval!(gm)

    learn_constraint!(gm)
    println("Approximation accuracies: ", accuracy(gm))

    # Solving of model
    @test_throws OCTException globalsolve(gm) # inaccuracy check in globalsolve.
    set_param(gm, :ignore_accuracy, true)
    globalsolve(gm);
    vals = solution(gm);
    println("X values: ", vals)
    println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

    # Testing constraint addition and removal
    clear_tree_constraints!(gm) # Clears all bbl constraints
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbls[2].mi_constraints])
    add_tree_constraints!(gm, gm.bbls[2])
    @test all([is_valid(gm.model, constraint) for constraint in gm.bbls[2].mi_constraints])
    add_tree_constraints!(gm);
    clear_tree_constraints!(gm, gm.bbls[1])
    @test !any(is_valid(gm.model, constraint) for constraint in gm.bbls[1].mi_constraints)
    clear_tree_constraints!(gm) # Finds and clears the one remaining bbl constraint.
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbls[1].mi_constraints])
    @test all([!is_valid(gm.model, var) for var in values(gm.bbls[1].leaf_variables)])

    # Saving fit for test_load_fits()
    save_fit(gm)

    # Testing finding bounds of bounded model
    @test isnothing(get_unbounds(gm.bbls))
    @test isnothing(find_bounds!(gm))

    # Testing clearing all data
    clear_data!(gm)
    @test all([size(bbl.X, 1) == 0 for bbl in gm.bbls])
end

test_expressions()

test_variables()

test_bounds()

test_sets()

test_linearize()

test_bbl()

test_kwargs()

test_basic_gm()
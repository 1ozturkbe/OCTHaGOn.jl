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
    expr_vars = vars_from_expr(expr, model)
    @test get_var_ranges(expr_vars) == [(1:5),(6:8),9]
    flat_expr = :((x...) -> $(expr)([x[i] for i in $(get_var_ranges(expr_vars))]...))
    fn = functionify(flat_expr)
    @test Base.invokelatest(fn, [1,2,3,4,1,5,-6,-7,7]...) == Base.invokelatest(f, ([1,2,3,4,1], [5,-6,-7], 7)...)
    @test Base.invokelatest(fn, flat(expr_vars)...) == res

    # Testing proper mapping for expressions
    flatvars = flat([y[2], z, x[1:4]])
    vars = vars_from_expr(expr, model)
    @test get_varmap(vars, flatvars) == [(2,2), (3,0), (1,1), (1, 2), (1,3), (1,4)]
    @test infarray([(1,4), (1,3), (2,0)]) == [[Inf, Inf, Inf, Inf], Inf]

    # Testing gradientify
    grad = gradientify(expr, vars_from_expr(expr, model))
    @test grad(ones(9)) == [1, 1, 1, 1, 0, -1, -1, 0, 1]
    other_grad = gradientify(simp_expr, vars_from_expr(simp_expr, model))
    @test other_grad(ones(5)) == 5 .* ones(5)
    @test grad(ones(9)) == [1, 1, 1, 1, 0, -1, -1, 0, 1] # in case of world age problems
    con = @constraint(model, Base.invokelatest(f, (x,y,z)...) >= 1) # also on JuMP constraints
    gradfn = gradientify(con, expr_vars)
    @test gradfn(ones(9)) == grad(ones(9))
    @test_throws OCTException gradientify(@constraint(model, [x[1] x[2];
                                                              x[3] x[4]] in PSDCone()), x[1:4])

    # Testing vars_from_constraint as well
    @test all([var in [x[1], x[2], x[3], x[4], y[1], y[2], z] for var in vars_from_constraint(con)])
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

function test_nonlinearize(gm::GlobalModel = minlp(true))
    nonlinearize!(gm)
    set_optimizer(gm, CPLEX_SILENT)
    @test_throws ErrorException("The solver does not support nonlinear problems (i.e., NLobjective and NLconstraint).") optimize!(gm)
    @test true
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
    bbls = [BlackBoxClassifier(constraint = nl_constr, vars = [x[4], x[5], z], expr_vars = [x[4], x[5], z]),
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
    learn_constraint!(bbl, true);

    # Check feasibility and accuracy
    @test 0 <= feasibility(bbl) <= 1
    @test 0 <= evaluate_accuracy(bbl) <= 1

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

""" Tests different kinds of regression. """
function test_regress()
    X = DataFrame(:x => 3*rand(100) .- 1, :y => 3*rand(100) .- 1);
    Y = Array(X[!,:x].^3 .* sin.(X[!,:y]));
    (α0, α), (β0, β), (γ0, γ) = ul_regress(X, Y)
    lowers = β0 .+ Matrix(X) * β;
    uppers = α0 .+ Matrix(X) * α;
    best_fit = γ0 .+ Matrix(X) * γ;
    @test all(lowers .<= Y) && all(uppers .>= Y) && all(uppers .>= lowers)
    errors = [sum((lowers-Y).^2), sum((uppers-Y).^2), sum((best_fit-Y).^2)]
    @test errors[3] <= errors[1] && errors[3] <= errors[2]  

    X = DataFrame(:x => rand(100), :y => rand(100))
    Y = X[!,:y] - X[!,:x] .+ 0.1
    solver = CPLEX_SILENT
    β0, β = svm(Matrix(X), Y)
    predictions = Matrix(X) * β .+ β0 
    @test sum((predictions-Y).^2) <= 1e-10
end

""" Tests various ways to train a regressor"""
function test_regressors()
    gm = minlp(true)
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    bbr = gm.bbls[3]

    # Threshold training
    learn_constraint!(bbr, threshold = 10)
    lnr = bbr.learners[end]
    @test lnr isa IAI.OptimalTreeClassifier
    all_leaves = find_leaves(lnr)
    # Add a binary variable for each leaf
    feas_leaves =
        [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    @test sort(collect(keys(bbr.ul_data[end]))) == sort(feas_leaves)
    @test bbr.thresholds[end] == 10

    # Check clearing and adding of tree constraints as well
    types = JuMP.list_of_constraint_types(gm.model)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    init_variables = length(all_variables(gm))
    add_tree_constraints!(gm, bbr)
    types = JuMP.list_of_constraint_types(gm.model)
    final_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    final_variables = length(all_variables(gm.model))
    @test final_constraints == init_constraints + length(all_mi_constraints(bbr)) + length(bbr.leaf_variables)    
    @test final_variables == init_variables + length(bbr.leaf_variables)
    clear_tree_constraints!(gm, bbr)
    types = JuMP.list_of_constraint_types(gm.model)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    @test init_variables == length(all_variables(gm))

    # Flat prediction training
    learn_constraint!(bbr, regression_sparsity = 0, max_depth = 2)
    lnr = bbr.learners[end]
    @test lnr isa IAI.OptimalTreeRegressor
    all_leaves = find_leaves(lnr)
    @test sort(collect(keys(bbr.ul_data[end]))) == sort(all_leaves)
    @test bbr.thresholds[end] == nothing

    # Full regression training
    learn_constraint!(bbr, max_depth = 1)
    lnr = bbr.learners[end]
    @test lnr isa IAI.OptimalTreeRegressor
    all_leaves = find_leaves(lnr)
    @test isempty(bbr.ul_data[end])
    @test isnothing(bbr.thresholds[end])

    # Checking proper storage
    @test all(length.([bbr.ul_data, bbr.thresholds, bbr.learners, bbr.learner_kwargs]) .== 3)
    clear_tree_data!(bbr)
    @test all(length.([bbr.ul_data, bbr.thresholds, bbr.learners, bbr.learner_kwargs]) .== 0)
end

""" Tests basic functionalities in GMs. """
function test_basic_gm()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    uniform_sample_and_eval!(gm)

    learn_constraint!(gm)
    println("Approximation accuracies: ", evaluate_accuracy(gm))

    # Solving of model
    set_param(gm, :ignore_accuracy, true)
    globalsolve(gm);
    vals = solution(gm);
    init_leaves = find_leaf_of_soln.(gm.bbls)
    println("X values: ", vals)
    println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

    # Testing constraint addition and removal
    clear_tree_constraints!(gm) # Clears all bbl constraints
    @test !any(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[2]))
    add_tree_constraints!(gm, gm.bbls[2])
    @test all(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[2]))
    add_tree_constraints!(gm);
    clear_tree_constraints!(gm, gm.bbls[1])
    @test !any(is_valid(gm.model, constraint) for constraint in all_mi_constraints(gm.bbls[1]))
    clear_tree_constraints!(gm) # Finds and clears the one remaining bbl constraint.
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbls[1].mi_constraints])
    @test all([!is_valid(gm.model, var) for var in values(gm.bbls[1].leaf_variables)])

    # Saving fit for test_load_fits()
    save_fit(gm)

    # Testing finding bounds of bounded model
    @test isnothing(get_unbounds(gm.bbls))
    @test isnothing(find_bounds!(gm))

    # Test reloading
    load_fit(gm)
    @test all([get_param(bbl, :reloaded) == true for bbl in gm.bbls])
    bbr = gm.bbls[1]
    @test bbr.thresholds[end] == bbr.thresholds[end-1]
    @test bbr.learner_kwargs[end] == bbr.learner_kwargs[end-1]
    bbc = gm.bbls[2]
    @test bbc.accuracies[end] == bbc.accuracies[end-1]
    globalsolve(gm)
    final_leaves = find_leaf_of_soln.(gm.bbls)
    @test all(init_leaves .== final_leaves)
    @test gm.solution_history[end, "obj"] ≈ gm.solution_history[end-1, "obj"] 

    # Testing clearing all data
    clear_data!(gm)
    @test all([size(bbl.X, 1) == 0 for bbl in gm.bbls])
    @test all([length(bbl.learners) == 0 for bbl in gm.bbls])
end

function test_gradients()
    gm = test_gqp()
    bbl = gm.bbls[1]
    set_param(bbl, :n_samples, 100)
    uniform_sample_and_eval!(gm)
    gradvals = evaluate_gradient(bbl, bbl.X)
    hand_calcs = [[6*x[1] + 2*x[2] + 1, 2*x[2] + 2*x[1] + 6] for x in eachrow(Matrix(bbl.X))]
    @test all(gradvals .== evaluate_gradient(bbl, Matrix(bbl.X)) .== hand_calcs)
    
    # Testing adding gradient cuts
    for i=1:size(bbl.X, 1)
        @constraint(gm.model, bbl.dependent_var >= sum(gradvals[i] .* (bbl.vars .- Array(bbl.X[i, :]))) + bbl.Y[i])
    end
    optimize!(gm)
    @test all(isapprox(Array(solution(gm))[i], [0.5, 1.0, 11.25][i], atol=0.1) for i=1:3)
end

test_expressions()

test_variables()

test_bounds()

test_sets()

test_linearize()

test_nonlinearize()

test_bbl()

test_kwargs()

test_regress()

test_regressors()

test_basic_gm()

test_gradients()

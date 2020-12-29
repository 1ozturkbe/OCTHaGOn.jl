#=
algorithms:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-27
=#

""" Tests basic functionalities in GMs. """
function test_basic_functions()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, GUROBI_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    bound!(gm, Dict(gm.vars[end] => [-300, 0]))
    sample_and_eval!(gm)
    sample_and_eval!(gm)

    learn_constraint!(gm)
    println("Approximation accuracies: ", accuracy(gm))

    # Solving of model
    @test_throws OCTException globalsolve(gm) # inaccuracy check in globalsolve.
    gm.settings[:ignore_accuracy] = true
    globalsolve(gm);
    vals = solution(gm);
    println("X values: ", vals)
    println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

    # Testing constraint addition and removal
    clear_tree_constraints!(gm) # Clears all BBF constraints
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbfs[2].mi_constraints])
    add_tree_constraints!(gm, [gm.bbfs[2]])
    @test all([is_valid(gm.model, constraint) for constraint in gm.bbfs[2].mi_constraints])
    add_tree_constraints!(gm);
    clear_tree_constraints!(gm, [gm.bbfs[1]])
    @test !any(is_valid(gm.model, constraint) for constraint in gm.bbfs[1].mi_constraints)
    clear_tree_constraints!(gm) # Finds and clears the one remaining BBF constraint.
    @test all([!is_valid(gm.model, constraint) for constraint in gm.bbfs[1].mi_constraints])
    @test all([!is_valid(gm.model, var) for var in gm.bbfs[1].leaf_variables])

    # Saving fit for test_load_fits()
    save_fit(gm)
    @test true
end

""" Tests loading of previously solved GMs.
NOTE: test_basic_functions MUST be called first. """
function test_load_fits(gm::GlobalModel = sagemark_to_GlobalModel(3; lse=false))
    set_optimizer(gm, GUROBI_SILENT);
    load_fit(gm);
    @test all([bbf.settings[:reloaded] == true for bbf in gm.bbfs])
    globalsolve(gm)
    @test true
end

function test_nonlinear_solve(gm::GlobalModel = GAMS_to_GlobalModel(OCT.GAMS_DIR, "problem3.13.gms"),
                              solver = IPOPT_SILENT)
    nonlinear_solve(gm, solver = solver)
    feas, infeas = evaluate_feasibility(gm)
    @test length(infeas) == 0
end

function test_find_bounds(gm::GlobalModel = GAMS_to_GlobalModel(OCT.GAMS_DIR, "problem3.13.gms"))
    set_optimizer(gm, GUROBI_SILENT)
    old_bounds = get_bounds(gm.bbfs)
    linear_bounds = find_bounds!(gm)
    @test true
end

test_basic_functions()

test_load_fits()

test_nonlinear_solve()

test_find_bounds()
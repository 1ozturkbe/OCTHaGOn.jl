""" Tests basic functionalities in GMs. """
function test_basic_functions()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    bound!(gm, Dict(gm.vars[end] => [-300, 0]))
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

    # Testing finding bounds of bounded model
    @test isnothing(get_unbounds(gm.bbfs))
    @test isnothing(find_bounds!(gm))
end

""" Tests loading of previously solved GMs.
NOTE: test_basic_functions MUST be called first. """
function test_load_fits()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT);
    load_fit(gm);
    @test all([get_param(bbf, :reloaded) == true for bbf in gm.bbfs])
    globalsolve(gm)
    @test true
end

function test_baron_solve(gm::JuMP.Model = gear(false))
    set_optimizer(gm, BARON_SILENT)
    optimize!(gm)
    @test true
end

function test_find_bounds(gm::GlobalModel = minlp(true))
    set_optimizer(gm, CPLEX_SILENT)
    old_bounds = get_bounds(gm.bbfs)
    linear_bounds = find_bounds!(gm)
    @test true
end

function test_speed_params(gm::GlobalModel = gear(true), solver = CPLEX_SILENT)
    gm = gear(true)
    solver = CPLEX_SILENT
    set_optimizer(gm, solver)   
    bound!(gm, gm.vars[end] => [-10,10]) 
    bbf = gm.bbfs[1]
    uniform_sample_and_eval!(bbf)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [5, 10]
    score_mat = [[], []]
    tree_mat = [[], []]
    time_mat = [[], []]
    for i=1:length(ls_num_hyper_restarts)
        for j=1:length(ls_num_tree_restarts)
            t1 = time()
            params = Dict(:ls_num_hyper_restarts => ls_num_hyper_restarts[i],
                          :ls_num_tree_restarts => ls_num_tree_restarts[j],
                          :max_depth => 2)
            learn_constraint!(bbf; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbf.learners[end].lnr)
            push!(score_mat[i], bbf.accuracies[end])
        end
    end
    @test true
end

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    find_bounds!(gm, all_bounds=true)
    set_param(gm, :ignore_feasibility, true)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    @info ("Sample feasibilities ", feasibility(gm))
    learn_constraint!(gm)
    @info("Approximation accuracies: ", accuracy(gm))
    save_fit(gm)
    globalsolve(gm)
    return
end

function test_recipe(gm::GlobalModel = minlp(true))
    recipe(gm)
    print(gm.solution_history)
    @test true
end

test_basic_functions()

test_load_fits()

test_baron_solve()

test_find_bounds()

test_speed_params()
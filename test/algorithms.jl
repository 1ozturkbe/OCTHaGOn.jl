""" Tests loading of previously solved GMs.
NOTE: test_basic_functions MUST be called first. """
function test_load_fits()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT);
    load_fit(gm);
    @test all([get_param(bbl, :reloaded) == true for bbl in gm.bbls])
    globalsolve(gm)
    @test true
end

function test_baron_solve(gm::JuMP.Model = gear(false))
    set_optimizer(gm, BARON_SILENT)
    optimize!(gm)
    sol = solution(gm)
    @test true
end

function test_find_bounds(gm::GlobalModel = minlp(true))
    set_optimizer(gm, CPLEX_SILENT)
    old_bounds = get_bounds(gm.bbls)
    linear_bounds = find_bounds!(gm)
    @test true
end

function test_speed_params(gm::GlobalModel = gear(true), solver = CPLEX_SILENT)
    gm = gear(true)
    solver = CPLEX_SILENT
    set_optimizer(gm, solver)   
    bound!(gm, gm.vars[end] => [-10,10]) 
    bbl = gm.bbls[1]
    uniform_sample_and_eval!(bbl)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [5, 10]
    tree_mat = [[], []]
    time_mat = [[], []]
    for i=1:length(ls_num_hyper_restarts)
        for j=1:length(ls_num_tree_restarts)
            t1 = time()
            params = Dict(:ls_num_hyper_restarts => ls_num_hyper_restarts[i],
                          :ls_num_tree_restarts => ls_num_tree_restarts[j],
                          :max_depth => 2)
            learn_constraint!(bbl; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbl.learners[end])
        end
    end
    @test true
end

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    find_bounds!(gm, all_bounds=false)
    set_param(gm, :ignore_feasibility, true)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    save_fit(gm)
    globalsolve(gm)
    return
end

function test_recipe(gm::GlobalModel = minlp(true))
    recipe(gm)
    print(gm.solution_history)
    @test true
end

function test_loaded_recipe(gm::GlobalModel = minlp(true))
    @info "GlobalModel " * gm.name * " reloaded..."
    set_optimizer(gm, CPLEX_SILENT)
    load_fit(gm)
    globalsolve(gm)
    @test true
end

test_load_fits()

test_baron_solve()

test_find_bounds()

test_speed_params()

test_recipe()

test_loaded_recipe()
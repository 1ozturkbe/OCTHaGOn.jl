function test_baron_solve(gm::JuMP.Model = gear(false))
    set_optimizer(gm, BARON_SILENT)
    optimize!(gm)
    sol = solution(gm)
    @test true
end

function test_speed_params(gm::GlobalModel = minlp(true), solver = CPLEX_SILENT)
    set_optimizer(gm, solver)   
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

test_baron_solve()

test_speed_params()
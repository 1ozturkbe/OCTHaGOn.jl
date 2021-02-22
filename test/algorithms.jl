function test_baron_solve(m::JuMP.Model = gear(false))
    set_optimizer(m, BARON_SILENT)
    optimize!(m)
    sol = solution(m)
    @test true
end

function test_speed_params(gm::GlobalModel = minlp(true), solver = CPLEX_SILENT)
    set_optimizer(gm, solver)   
    bbl = gm.bbls[1]
    uniform_sample_and_eval!(bbl)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [3, 5]
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

function test_classify_gradients()
    gm = minlp(true)
    bbr = gm.bbls[3]
    uniform_sample_and_eval!(gm)
    idxs = collect(1:10)
    classify_curvature(bbr, idxs)
    @test !any(ismissing.(bbr.curvatures[idxs]))
    classify_curvature(bbr)
    @test all(bbr.curvatures .> 0)
    update_vexity(bbr)
    @test bbr.convex == true
    @test bbr.local_convexity == 1.
end

function test_infeasibility_cuts()
    gm = sagemark_to_GlobalModel(15, lse=false)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    add_tree_constraints!(gm)
    optimize!(gm)
    bbc_idxs = [bbl isa BlackBoxClassifier for bbl in gm.bbls]
    while any(gm.feas_history[end] .* bbc_idxs .!= 0)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_feasibility_sample()
    gm = speed_reducer()
    uniform_sample_and_eval!(gm)
    @test any(check_feasibility(gm) .!= 1)
    feasibility_sample(gm)
    @test all(check_feasibility(gm) .== 1)
end

function test_survey_method(gm::GlobalModel = minlp(true))
    gm = minlp(true)
    uniform_sample_and_eval!(gm)
    surveysolve(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    bbc_idxs = [x isa BlackBoxClassifier for x in gm.bbls]
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        update_vexity.(bbrs)
    end
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while gm.cost[end] > gm.cost[end-1] 
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
end

test_baron_solve()

test_speed_params()

test_classify_gradients()

test_infeasibility_cuts()

test_feasibility_sample()

test_survey_method()
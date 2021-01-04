""" Tests basic functionalities in GMs. """
function test_basic_functions()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, CPLEX_SILENT)

    # Actually trying to optimize...
    find_bounds!(gm, all_bounds=true)
    bound!(gm, Dict(gm.vars[end] => [-300, 0]))
    sample_and_eval!(gm)
    sample_and_eval!(gm)

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

# function test_relaxations(gm::GlobalModel = minlp(true),
#                           solver = CPLEX_SILENT)
#     gm = minlp(true)
#     set_optimizer(gm, solver)
#     bound!(gm, gm.vars[end] => [-10,20])
#     sample_and_eval!(gm)
#     sample_and_eval!(gm)
#     # Separate by percentiles of infeasible
#     percentiles = [0.75, 0.9, 0.98, 1.0]
#     bbf = gm.bbfs[1]
#     for i=1:length(thresholds)
#         learn_constraint!(bbf, validation_criterion = :sensitivity, threshold = thresholds[i])
#     end
#     @test true
# end

function test_speed_params(gm::GlobalModel = gear(true), solver = CPLEX_SILENT)
    gm = gear(true)
    solver = CPLEX_SILENT
    set_optimizer(gm, solver)   
    bound!(gm, gm.vars[end] => [-10,10]) 
    bbf = gm.bbfs[1]
    sample_and_eval!(bbf)
    sample_and_eval!(bbf)
    sample_and_eval!(bbf)    
    
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
                          :ls_num_tree_restarts => ls_num_tree_restarts[j])
            learn_constraint!(bbf; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbf.learners[end].lnr)
            push!(score_mat[i], bbf.accuracies[end])
        end
    end
    @test true
    # Sensitivity parameters
    # feas_ratio = feasibility(bbf)
    # thresholds = [0.6, 0.85, 0.98]
    # trees = []
    # scores = []
    # times = []
    # sens = []
    # for i = 1:length(thresholds)
    #     t1 = time()
    #     learn_constraint!(bbf, validation_criterion = :sensitivity, threshold = thresholds[i])
    #     push!(times, time()-t1)
    #     push!(trees, bbf.learners[end].lnr)
    #     push!(scores, bbf.accuracies[end])
    #     push!(sens, )    
    # end
end

test_basic_functions()

test_load_fits()

test_baron_solve()

test_find_bounds()

# test_relaxations()

test_speed_params()

# gm = minlp(true)
# bounds = get_bounds(gm)
# lin_bounds = find_linear_bounds!(gm)
# unboundeds = get_unbounds(gm)
# unbounded_vars = collect(keys(unboundeds))
# matching = match_bbfs_to_vars(gm, unbounded_vars);

# M=1e3
# bbf = matching[1].first
# # function exponential_bound(bbf::BlackBoxFunction, M = max_iterations::Int64 = 3)
#     unbounds = get_unbounds(bbf)
#     for (var, bounds) in unbounds
#         JuMP.set_lower_bound(var, 0)
#         JuMP.set_upper_bound(var, 1)
#     end
#     df = boundary_sample(bbf, fraction = 0.5)
#     append!(df, lh_sample(bbf, iterations = 1, n_samples = get_param(bbf, :n_samples) - size(df, 1)), cols=:setequal)
#     for (var, bounds) in unbounds # Revert bounds
#         !isinf(minimum(bounds)) && JuMP.set_lower_bound(var, minimum(bounds))
#         !isinf(maximum(bounds)) && JuMP.set_upper_bound(var, maximum(bounds))
#         if all(isinf.(bounds)) # Log-scale the samples
#             df[!,string(var)] = 2. .* (rand(get_param(bbf, :n_samples)) .- 0.5) .* M.^ df[!,string(var)]
#         elseif isinf(minimum(bounds))
#             df[!,string(var)] = maximum(bounds) - M.^ df[!,string(var)]
#         elseif isinf(maximum(bounds))
#             df[!,string(var)] = minimum(bounds) - M.^ df[!,string(var)]
#         end
#     end
#     eval!(bbf, df)
#     df = knn_sample(bbf)
#     eval!(bbf, df)
#     df = knn_sample(bbf)
#     eval!(bbf, df)
#     feas_X = bbf.X[findall(x -> x .>= 0, bbf.Y),:];
#             # Find tightest enclosing box
#             box_bounds = Dict(bbf.vars[i] => [minimum(feas_X[!,i]), maximum(feas_X[!,i])] for i=1:length(bbf.vars))
# #         end
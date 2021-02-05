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

""" 
    classify_curvature(bbr::BlackBoxRegressor)

Classify curvature of KNN patches. 
"""
function classify_curvature(bbr::BlackBoxRegressor, idxs = 1:size(bbr.X, 1))
    build_knn_tree(bbr);
    idxs, dists = find_knn(bbr, k=length(bbr.vars) + 1);
    X = Matrix(bbr.X)
    curvs = zeros(length(idxs))

    for i in idxs
        diffs = [Array(bbr.X[i, :]) - Array(bbr.X[j, :]) for j in idxs[i]]
        center_grad = bbr.gradients[i]
        under_offsets = [-dot(center_grad, differ) for differ in diffs]
        actual_offsets = bbr.Y[idxs[i]] .- bbr.Y[i]
        if all(actual_offsets .>= under_offsets .- 1e-10)
            append!(up_idxs, i)
        elseif all(actual_offsets .<= under_offsets .- 1e-10)
            append!(down_idxs, i)
        else
            append!(mixed_idxs, i)
        end
    end
end

function test_classify_gradients()
    gm = gear(true)
    bbr = gm.bbls[1]
    uniform_sample_and_eval!(gm)
    # learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6)
    # update_tree_constraints!(gm, bbr)
    # optimize!(gm)


    @test true
end

test_baron_solve()

test_speed_params()

test_classify_gradients()
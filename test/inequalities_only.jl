function test_refinement()
    gm = speed_reducer()
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)

    uniform_sample_and_eval!(gm)
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier
            learn_constraint!(bbl)
        else
            learn_constraint!(bbl, regression_sparsity = 0)
        end
        add_tree_constraints!(gm, bbl)
    end
    optimize!(gm)
    # println("Samples: $(size(gm.bbls[1].X, 1))")

    gap = gm.feas_history[end]

    for idx in findall(x -> x != 0, gap)
        if gm.bbls[idx] isa BlackBoxClassifier
            bbl = gm.bbls[idx]
            vars = bbl.vars;
            # scatter(bbl.X[:, string(vars[1])], bbl.X[:, string(vars[2])], color = :blue)
            last_lnr = bbl.learners[end];
            last_sample = bbl.X[end, :];
            last_value = bbl.Y[end];
            leaves = IAI.apply(last_lnr, bbl.X);
            leaf_neighbors = findall(x -> x == leaves[end], leaves);

            # Building KNN tree    
            # df = knn_sample(bbl, k = length(bbl.vars), tighttol = get_param(gm, :tighttol), sample_idxs = leaf_neighbors)
            # last_sample = bbl.knn_tree.data[end]
            # println("New_samples: ", size(df))
            # if size(df, 1) > 0
            #     eval!(bbl, df)
            # #     weights = reweight(normalized_data(bbl))
            # #     learn_constraint!(bbl, sample_weight = weights)
            #     clear_tree_constraints!(gm, bbl)
            #     add_tree_constraints!(gm, bbl)
            # #     scatter!(df[:, string(vars[1])], df[:, string(vars[2])], color = :red)
            # end
            # LEARN TREE ON DF AS WELL AS ORIGINAL DATA IN THE LEAF. 
            # REMOVE ORIGINAL CONSTRAINT, RETRAIN WITH THE NEW DATA
            # weights = reweight(normalized_data(bbl), mag)
            # Learn greedy SVM approx locally
            # samp = 30
            # idxs, dists = OCT.knn(bbl.knn_tree, [last_sample], samp)
            # β0, β = svm(Matrix(bbl.X[idxs[1], :]), bbl.Y[idxs[1]])
            leaf_var = bbl.leaf_variables[leaves[end]]
            M = 1e5
            gfn = x -> ForwardDiff.gradient(bbl.f, x)
            varmap = get_varmap(bbl.expr_vars, bbl.vars)        
            inp = deconstruct(solution(gm), bbl.vars, bbl.expr_vars, varmap)
            grad = gfn(inp[1]...)    
            grad = [grad[v[2]] for v in varmap]
            constr = @constraint(gm.model, 0 <= sum(grad .* bbl.vars) + M * (1 - leaf_var))
            push!(bbl.mi_constraints[leaves[end]], constr)
        end
    end
    optimize!(gm)
    @test true
end

test_refinement()

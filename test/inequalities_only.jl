function test_refinement()
    gm = nlp2(true)
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)

    uniform_sample_and_eval!(gm)
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier
            learn_constraint!(bbl, get_param(gm, :ignore_feasibility))
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
            scatter(bbl.X[:, string(vars[1])], bbl.X[:, string(vars[2])], color = :blue)
            last_lnr = bbl.learners[end];
            last_sample = bbl.X[end, :];
            last_value = bbl.Y[end];
            leaves = IAI.apply(last_lnr, bbl.X);
            leaf_neighbors = findall(x -> x == leaves[end], leaves);

            # Building KNN tree    
            df = knn_sample(bbl, k = length(bbl.vars), tighttol = get_param(gm, :tighttol), sample_idxs = leaf_neighbors)
            println("New_samples: ", size(df))
            # LEARN TREE ON DF AS WELL AS ORIGINAL DATA IN THE LEAF. 
            # REMOVE ORIGINAL CONSTRAINT, RETRAIN WITH THE NEW DATA
            if size(df, 1) > 0
                eval!(bbl, df)
                weights = reweight(normalized_data(bbl))
                learn_constraint!(bbl, sample_weight = weights)
                clear_tree_constraints!(gm, bbl)
                add_tree_constraints!(gm, bbl)
                scatter!(df[:, string(vars[1])], df[:, string(vars[2])], color = :red)
            end
        end
    end
    optimize!(gm)
    @test true
end

test_refinement()

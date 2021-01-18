using Plots

gm = speed_reducer()
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
        leaf_sign = sign.(bbl.Y[leaf_neighbors]);

        # Building KNN tree    
        df = DataFrame([Float64 for i in string.(bbl.vars)], string.(bbl.vars));
        build_knn_tree(bbl);
        idxs, dists = find_knn(bbl);
        # Relevant idxs
        idxs = idxs[leaf_neighbors];
        dists = dists[leaf_neighbors];
        feas_class = classify_patches(bbl, idxs);
        for i=1:length(leaf_neighbors) # This loop is for making sure that every possible root is sampled only once.
            if feas_class[i] == "mixed" && leaf_sign[i] == 1
                nodes = [leaf_neighbors[i]]
                for j = 1:length(idxs[i])
                    if sign(bbl.Y[idxs[i][j]]) != leaf_sign[i] && dists[i][j] >= get_param(gm, :tighttol)
                        push!(nodes, j)
                    end
                end
                np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
                append!(df, np);
            end
        end
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

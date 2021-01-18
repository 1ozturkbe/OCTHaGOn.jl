# using Plots

gm = nlp2(true)
set_param(gm, :ignore_accuracy, true)

uniform_sample_and_eval!(gm)
learn_constraint!(gm)
add_tree_constraints!(gm)
optimize!(gm)
gap = feas_gap(gm)
# println("Samples: $(size(gm.bbls[1].X, 1))")


for idx in findall(x -> x != 0, gap)
    # scatter(gm.bbls[1].X[:,"x[1]"], gm.bbls[1].X[:,"x[2]"], color = :blue)
    bbl = gm.bbls[idx]
    last_lnr = bbl.learners[end]
    last_sample = bbl.X[end, :]
    last_value = bbl.Y[end]
    leaves = IAI.apply(last_lnr, bbl.X)
    last_leaf = leaves[end]
    leaf_neighbors = findall(x -> x == leaves[end], leaves)
    leaf_sign = sign.(bbl.Y[leaf_neighbors])

    # Building KNN tree    
    df = DataFrame([Float64 for i in string.(bbl.vars)], string.(bbl.vars))
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl);
    # Relevant idxs
    idxs = idxs[leaf_neighbors]
    dists = dists[leaf_neighbors]
    feas_class = classify_patches(bbl, idxs);
    for i=1:length(leaf_neighbors) # This loop is for making sure that every possible root is sampled only once.
        if feas_class[i] == "mixed"
            nodes = [idx for idx in idxs[i] if sign(bbl.Y[idx]) != leaf_sign[i]];
            push!(nodes, leaf_neighbors[i])
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np);
        end
    end
    # LEARN TREE ON DF AS WELL AS ORIGINAL DATA IN THE LEAF. 
    # REMOVE ORIGINAL CONSTRAINT, RETRAIN WITH THE NEW DATA
    eval!(bbl, df)
    weights = reweight(Matrix(bbl.X), Array(last_sample))
    learn_constraint!(bbl, sample_weight = weights)
    clear_tree_constraints!(gm, bbl)
    add_tree_constraints!(gm, bbl)
    # scatter!(df[:,"x[1]"], df[:,"x[2]"], color=:red)
end

optimize!(gm)
gap = feas_gap(gm)
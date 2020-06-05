using LatinHypercubeSampling
include("../src/fit.jl");
include("../src/constraintify.jl")
include("examples.jl");

function compile_lse_constraints()
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    ineq_constraints = Function[];
    convex = Bool[];
    linear = Bool[];
    idxs = Int64[];
    for idx=1:25
        fn_model = import_sagebenchmark(idx, lse = true);
        if !any(ismissing.(fn_model.lbs)) && !any(ismissing.(fn_model.ubs))
            println("Adding constraints from test ", string(idx))
            signomials, solver, run_fn = sagemarks.get_example(idx);
            f, greaters, equals = signomials;
            xdim = size(f.alpha,2);
            # Only testing inequality constraints for now...
            for j in greaters
                if abs(sum(j.alpha)) != 1.
                    append!(idxs, [idx])
                    append!(ineq_constraints, [alphac_to_fn(j.alpha, j.c, lse = true)])
                    if sum(float(j.c .>= zeros(length(j.c)))) == 1
                        append!(convex, [true])
                    else
                        append!(convex, [false])
                    end
                    if sum(float(j.c .>= zeros(length(j.c)))) == 1 && length(j.c) == 2
                        append!(linear, [true])
                    else
                        append!(linear, [false])
                    end
                end
            end
        end
    end
    println("Compiled all multivariate LSE inequalities!")
    return ineq_constraints, convex, linear, problem_idxs
end

ineqs, convex, linear, problem_idxs = compile_lse_constraints();
n_samples = 1000;

# Training base trees
lnr = base_otc()
lnr_grid = base_grid(base_otc())
for i=1:length(ineqs)
    print("Fitting constraint ", i, ".")
    fn_model = import_sagebenchmark(problem_idxs[i], lse=true)
    n_dims = length(fn_model.lbs);
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    feas_tree = learn_constraints!(lnr, [ineqs[i]], X)
    IAI.write_json(string("data/constraint",i,".tree"), feas_tree[1])
end

# Loading the trees
grids = [IAI.read_json(string("data/constraint",i,".tree")) for i=1:length(ineqs)];
scores = [IAI.get_grid_results(grid)[1, 3] for grid in grids];
convex_idxs = findall(x -> x.> 0, convex)
nonconvex_idxs =  findall(x -> x.== 0, convex)
linear_idxs =  findall(x -> x.> 0, linear)

# Plotting all accuracies
# using Plots
# data = Dict("x" => idxs, "y" => scores)
# convex_scores = bar(convex_idxs, data["y"][convex_idxs], xlabel="Constraint number",
#                     xticks=1:length(ineqs), label="convex")
# nonconvex_scores = bar!(nonconvex_idxs, data["y"][nonconvex_idxs], xlabel="Constraint number",
#                         title="Accuracy over all constraints", xticks=1:length(ineqs),
#                         label="non-convex", legend=:bottomright)

# Comparing accuracy vs. convexity
# using Plots
# data = Dict(
#     "x" => ["linear", "convex", "non-convex"],
#     "y" => [sum(scores.*linear)/sum(linear),
#             sum(scores.*convex)/sum(convex),
#             sum(scores.*(1 .- convex))/(length(convex) - sum(convex))],
# )
# convexity_effect = bar(data["x"], data["y"], title="Accuracy vs. convexity", legend=false)

# Retrain non-convex with different numbers of samples
n_samples = [100, 250, 500, 750, 1000];
for idx in nonconvex_idxs
    fn_model = import_sagebenchmark(idxs[idx], lse=true);
    n_dims = length(fn_model.lbs);
    for j in n_samples
        plan, _ = LHCoptim(j, n_dims, 1);
        X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
        feas_tree = learn_constraints!(lnr, [ineqs[idx]], X);
    IAI.write_json(string("data/constraint",idx,"_",j,"samples",".tree"), feas_tree[1]);
    end
end

# Loading trees, and scoring w.r.t. maximum of the samples
grids = [IAI.read_json(string("data/constraint",idx,"_",j,"samples",".tree")) for idx in nonconvex_idxs
         for j in n_samples];
scores = [];
for idx in nonconvex_idxs
    fn_model = import_sagebenchmark(idxs[idx], lse=true);
    n_dims = length(fn_model.lbs);
    plan, _ = LHCoptim(maximum(n_samples), n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    Y = [ineqs[idx](X[j,:]) >= 0 for j = 1:maximum(n_samples)];
    score_list = [];
    for j in n_samples
        nlnr = IAI.read_json(string("data/constraint",idx,"_",j,"samples",".tree"));
        append!(score_list, IAI.score(nlnr.lnr, X, Y))
    end
    append!(scores, score_list)
end
scores = reshape(scores, (length(nonconvex_idxs), length(n_samples)));
# Finally plot
for i=1:length(nonconvex_idxs)
    display(plot!(scores[i,:]))
end

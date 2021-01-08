#=
test_src:
- Author: Berk
- Date: 2020-06-16
This tests lse approximations.
=#

using PyCall

function compile_lse_constraints(n_samples = 300)
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    ineq_constraints = BlackBoxLearner[]
    for idx=1:25
        md = OCT.sagemark_to_GlobalModel(idx, lse = false);
        update_bounds!(md, lbs = Dict(md.vks[end] => -200), ubs= Dict(md.vks[end] => 200))
        if !any(isinf.(values(md.lbs))) && !any(isinf.(values(md.ubs)))
            println("Adding constraints from test ", string(idx));
            for i=1:length(md.bbls)
                set_param(md.bbls[i], :n_samples, n_samples)
                md.bbls[i].name = Float64(idx) + i/10.;
                push!(ineq_constraints, md.bbls[i]);
            end
        end
    end
    println("Compiled all multivariate LSE inequalities!")
    return ineq_constraints
end

ineqs = compile_lse_constraints();
uniform_sample_and_eval!(ineqs);

learn_constraint!(ineqs)

# Plotting all accuracies (non-convex approximation)
ineqs = ineqs[findall(x -> length(x.accuracies) > 0, ineqs)]
plot_accuracies(ineqs)

convex_idxs = findall(x -> "logconvex" in x.tags, ineqs);
nonconvex_idxs = findall(x -> x ∉ convex_idxs, [1:length(ineqs)]);

plot_accuracies(ineqs[convex_idxs])
plot_accuracies(ineqs[nonconvex_idxs])

# nonconvex_scores = bar!(nonconvex_idxs, [ineq.accuracies[end] for ineq in ineqs[nonconvex_idxs]],
#                   xlabel="Constraint number",
#                         title="Accuracy over all constraints", xticks=1:length(ineqs),
#                         label="non-convex", legend=:topright, ylims=[0.9, 1.])

# # Comparing accuracy vs. convexity
# data = Dict(
#     "x" => ["convex", "non-convex"],
#     "y" => [sum(scores.*convex)/sum(convex),
#             sum(scores.*(1 .- convex))/(length(convex) - sum(convex))],
# );
# convexity_effect = bar(data["x"], data["y"], title="Accuracy vs. convexity", legend=false,ylims=[0.9, 1.])
#
# # Retrain non-convex with different numbers of samples
# n_samples = [100, 250, 500, 750, 1000];
# for idx in problem_idxs
#     md = OCT.sagemark_to_GlobalModel(problem_idxs[idx], lse=false);
#     n_dims = length(md.lbs);
#     for j in n_samples
#         plan, _ = LHCoptim(j, n_dims, 1);
#         X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
#         feas_tree = learn_constraints!(lnr, [ineqs[idx]], X);
#     IAI.write_json(string(PROJECT_ROOT, "/test/data/constraint",idx,"_",j,"samples",".tree"), feas_tree[1]);
#     end
# end
#
# # Loading trees, and scoring w.r.t. maximum of the samples
# n_samples = [100, 250, 500, 750, 1000];
# grids = [IAI.read_json(string("data/constraint",idx,"_",j,"samples",".tree")) for j in n_samples for idx=1:length(ineqs)]; # inside for loop gets executed first.
# scores = [];
# for idx=1:length(ineqs)
#     md = OCT.sagemark_to_GlobalModel(problem_idxs[idx], lse=false);
#     n_dims = length(md.lbs);
#     plan, _ = LHCoptim(maximum(n_samples), n_dims, 1);
#     X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
#     Y = [ineqs[idx](X[j,:]) >= 0 for j = 1:maximum(n_samples)];
#     score_list = [];
#     for j in n_samples
#         nlnr = IAI.read_json(string("data/constraint",idx,"_",j,"samples",".tree"));
#         append!(score_list, IAI.score(nlnr, X, Y))
#     end
#     append!(scores, score_list)
# end
# scores = transpose(reshape(scores, (length(n_samples), length(ineqs))));
#
# # Finally plot
# for i=1:length(convex_idxs)
#     display(plot!(n_samples, scores[i,:], xlabel="Number of samples", ylabel="Accuracy",
#                   legend=false, color=:blue))
# end
# # Plotting trend
# display(plot!(n_samples, transpose(sum(scores, dims=1))./size(scores,1), xlabel="Number of samples", ylabel="Accuracy",
#                   legend=false,linewidth=4, color=:blue))
#
# # Dimension vs. accuracy
# plot(xlabel="Number of samples", ylabel="Accuracy")
# for i in unique(dimension)
#     dim_idxs = findall(x -> x==i, dimension);
#     y = sum(scores[dim_idxs, :], dims=1)./size(dim_idxs);
#     display(plot!(n_samples, y', label=string("Dimension ",i)))
# end
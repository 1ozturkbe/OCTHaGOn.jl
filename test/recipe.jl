# """ Very coarse solution for gm to check for feasibility and an initial starting point. """
# function surveysolve(gm::GlobalModel)
#     bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
#     learn_constraint!(bbcs)
#     bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
#     bbr = bbrs[1]
#     learn_constraint!(bbr, regression_sparsity = 0, max_depth = 2)
#     ul_data = ul_boundify(bbr.learners[end], bbr.X, bbr.Y)
#     # Setting an extremely conservative upper bound for the objective. 
#     # l_con = @constraint(gm.model, gm.model[:obj] >= α0 + sum(α .* bbr.vars))
#     for bbc in bbcs
#         add_tree_constraints!(gm, bbc)
#     end
#     add_tree_constraints!(gm, bbr)
#     optimize!(gm)
#     return
# end

# Set threshold using previous solution, and continue
gm = minlp(true)
set_optimizer(gm, CPLEX_SILENT)
uniform_sample_and_eval!(gm)
# bbr = gm.bbls[3]
# surveysolve(gm)

for bbl in gm.bbls
    if bbl isa IAI.OptimalTreeClassifier
        learn_constraint!(bbl)
    else
        learn_constraint!(bbl, )

threshold = getvalue(gm.model[:obj])
feasibility = evaluate_feasibility(gm)

# Changing the upper/lower bounds into a tree constraint. 
lnr = base_classifier()
# learn_constraint!(bbr; regression_sparsity = 0)
lnr = learn_from_data!(bbr.X, bbr.Y .<= threshold, lnr::IAI.OptimalTreeLearner)

bbr.mi_constraints, bbr.leaf_variables = add_feas_constraints!(gm.model, bbr.vars, lnr;
                                            M=M, equality = bbr.equality);





lnr = base_classifier()
IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
threshold = quantile(bbr.Y, 1-interval)
nl = learn_from_data!(bbr.X, bbr.Y .<= threshold, lnr; fit_classifier_kwargs(; kwargs...)...) 
all_leaves = find_leaves(nl)
feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
leaf_idx = IAI.apply(nl, bbr.X)
upper, lower = trust_region_data(lnr, Symbol.(bbr.vars))
for leaf in feas_leaves
    idx = findall(x -> x == leaf, leaf_idx)
    Y_ub = maximum(bbr.Y[idx])
    Y_lb = minimum(bbr.Y[idx])
    β0, β = ridge_regress(X[idx,:], Y[idx])
    predictions = β0 .+ Matrix(X[idx, :]) * β
    mse_ridge = sum(((predictions - Y[idx])./Y[idx]).^2)/length(idx)
    mse_tree = sum(((IAI.predict(bbr.learners[end], bbr.X[idx,:]) - Y[idx])./Y[idx]).^2)/length(idx)
    println(IAI.score(bbr.learners[end], bbr.X[idx, :], bbr.Y[idx]))
end

total_mse = sum(((IAI.predict(bbr.learners[end], bbr.X) - bbr.Y)./bbr.Y).^2)/length(bbr.Y)

        
push!(bbr.learners, nl);
bbr.predictions = IAI.predict(nl, bbr.X)
push!(bbr.learner_kwargs, Dict(kwargs))
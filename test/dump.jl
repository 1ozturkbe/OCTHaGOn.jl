# function example1_infeas()
#     n_samples = 1000;
#     n_dims = 3;
#     # Code for initial tree training
#     plan, _ = LHCoptim(n_samples, n_dims, 1);
#     X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:3]);
#     # Assuming the objective has already been trained...
#     lnr = base_otr()
#     name = "example1"
#     feasTrees = learn_constraints!(lnr, constraints, X, name=name)
#
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_infeas.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(Gurobi.Optimizer);
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_feas_constraints(constr, m, x, vks, 1000);
#     add_regr_constraints(objectivefn, m, x, obj, vks, 1000000);
#     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
#     status = optimize!(m)
#     println("Solved minimum: ", getvalue.(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue.(x))
#     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
# end

# m = Model(solver=GurobiSolver());
# @variable(m, x[1:n_dims])
# @variable(m, obj)
# @objective(m, Min, obj)
# for i=1:length(ineq_trees)
#     add_feas_constraints(ineq_trees[i], m, x, vks; M = 100);
# end
# for i=1:length(eq_trees)
#     add_feas_constraints(eq_trees[i], m, x, vks; M = 100);
# end
#
# #     add_regr_constraints(constr, m, x, 0, vks, 1000000);
# #     add_regr_constraints(objectivefn, m, x, obj, vks, 1000000);
# #     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
# #     status = optimize!(m)
# #     println("Solved minimum: ", getvalue.(obj))
# #     println("Known global bound: ", -147-2/3)
# #     println("X values: ", getvalue.(x))
# #     println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710])
#
# n_samples, n_features = size(X)
# objective = fn_model.obj
# Y = [objective(transpose(X[j, :])) for j = 1:n_samples]
# nX = repeat(X, n_samples)
# nY = repeat(Y, n_samples)
# nY = [nY[j] >= objective(nX[j,:]) for j=1:shape(nY,2)]
#     # Making sure that we only consider relevant features.
#     if !isnothing(idxs)
#         IAI.set_params!(lnr, split_features = idxs[i])
#         if typeof(lnr) == IAI.OptimalTreeRegressor
#             IAI.set_params!(lnr, regression_features=idxs[i])
#         end
#     else
#         IAI.set_params!(lnr, split_features = :all)
#         if typeof(lnr) == IAI.OptimalTreeRegressor
#             IAI.set_params!(lnr, regression_features=:all)
#         end
#     end
#     IAI.fit!(lnr, nX, nY)
#     return lnr

# function example_optimize!(md::ModelData; M=1e5)
#     """ Solves an already fitted function_model. """
#     # Retrieving constraints
#     obj_tree = IAI.read_json(string("data/", md.name, "_obj.json"));
#     ineq_trees = [IAI.read_json(string("data/", md.name, "_ineq_", i, ".json")) for i=1:length(md.ineqs)];
#     # Creating JuMP model
#     m = Model(solver=GurobiSolver());
#     n_vars = length(md.lbs);
#     vks = [Symbol("x",i) for i=1:n_vars];
#     obj_vks = [Symbol("x",i) for i=1:n_vars+1];
#     @variable(m, x[1:n_vars])
#     @variable(m, y)
#     @objective(m, Min, y)
#     add_feas_constraints!(obj_tree, m, [x; y], obj_vks; M=M)
#     for tree in ineq_trees
#         add_feas_constraints!(tree, m, x, vks; M=M)
#     end
#     for tree in eq_trees
#         add_feas_constraints!(tree, m, x, vks; M=M)
#     end
#     for i=1:n_vars # Bounding
#         @constraint(m, x[i] <= md.ubs[i])
#         @constraint(m, x[i] >= md.lbs[i])
#     end
#     status = optimize!(m)
#     return true
# end

# Sampling based on GaussianProcesses
#         plan = randomLHC(Int(round(bbf.n_samples*ratio)), n_dims);
#         random_samples = scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i in vks]);
#         μ, σ = predict_f(bbf.gp, random_samples');
#         cdf_0 = [Distributions.cdf(Distributions.Normal(μ[i], σ[i]),0) for i=1:size(random_samples,1)];
#          #TODO: add criterion for information as well (something like sortperm(σ))
#         # Sample places with high probability of being near boundary (0),
#         # but also balance feasibility ratio.
#         p = bbf.feas_ratio
# #         balance_fn = x -> -1*tan(2*atan(-0.5)*(x-0.5)) + 0.5
#         balance_fn = x -> -1/0.5^2*(x-0.5)^3 + 0.5;
#         indices = sortperm(abs.(cdf_0 .- balance_fn(p)));
#         samples = DataFrame(random_samples[indices[1:bbf.n_samples],:], vks);
#         eval!(bbf, samples);

# function predict(bbf::BlackBoxFunction, X::AbstractArray)
#     μ, σ = predict_f(bbf.gp, transpose(X))
#     return μ, σ
# end
#
# function optimize_gp!(bbf::BlackBoxFunction)
#     """ Optimizes a GaussianProcess over a BlackBoxFunction,
#     with adaptively changing kernel. """
# #         bbf.gp = ElasticGPE(length(bbf.idxs), # data
# #         mean = MeanConst(sum(bbf.Y)/length(bbf.Y)), logNoise = -10)
#     lbs = [bbf.lbs[key] for key in bbf.vks];
#     ubs = [bbf.ubs[key] for key in bbf.vks];
#     bbf.gp = GPE(transpose(Array(bbf.X)), bbf.Y, # data
#     MeanConst(sum(bbf.Y)/length(bbf.Y)),
#     SEArd(log.((ubs-lbs)./(2*sqrt(length(bbf.Y)))), -5.))
#     optimize!(bbf.gp); #TODO: optimize GP
#                        # Instead of regenerating at every run, figure out
#                        # how to update.
# end

# function plot(bbf::BlackBoxFunction)
#     """ Plots a BlackBoxFunction, which is either a 2D scatter plot or a 2D Gaussian Process. """
#     if !isnothing(bbf.gp)
#         if bbf.gp.dim > 1
#             throw(OCTException("plot(BlackBoxFunction) only works in 2D."))
#         end
#         Plots.plot(bbf.gp, legend=false, fmt=:png)
#     else
#         if size(bbf.X, 2) > 1
#             throw(OCTException("plot(BlackBoxFunction) only works in 2D."))
#         end
#         scatter(Matrix(bbf.X), bbf.Y, legend=false, fmt=:png)
#     end
# end
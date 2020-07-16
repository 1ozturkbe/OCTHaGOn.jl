# function example1_infeas()
#     n_samples = 1000;
#     n_dims = 3;
#     # Code for initial tree training
#     plan, _ = LHCoptim(n_samples, n_dims, 1);
#     X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:3]);
#     # Assuming the objective has already been trained...
#     lnr = base_otr()
#     name = "example1"
#     feasTrees = learn_constraints(lnr, constraints, X, name=name)
#
#     # Creating the model
#     constr = IAI.read_json("data/example1_constraint_infeas.json")
#     objectivefn = IAI.read_json("data/example1_objective_naive.json")
#     vks = [Symbol("x",i) for i=1:3];
#     m = Model(solver=GurobiSolver());
#     @variable(m, x[1:3])
#     @variable(m, obj)
#     @objective(m, Min, obj)
#     add_feas_constraints(constr, m, x, vks, 1000);
#     add_regr_constraints(objectivefn, m, x, obj, vks, 1000000);
#     bound_variables(m, x, fn_model.lbs, fn_model.ubs);
#     status = solve(m)
#     println("Solved minimum: ", getvalue(obj))
#     println("Known global bound: ", -147-2/3)
#     println("X values: ", getvalue(x))
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
# #     status = solve(m)
# #     println("Solved minimum: ", getvalue(obj))
# #     println("Known global bound: ", -147-2/3)
# #     println("X values: ", getvalue(x))
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

# TODO: turn into ModelData_solve
# function example_solve(md::ModelData; M=1e5)
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
#     status = solve(m)
#     return true
# end

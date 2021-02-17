# function add_convex_callback(gm::GlobalModel, bbc::BlackBoxClassifier, tighttol = get_param(gm, :tighttol))
#     cvx_cb = let bbc = bbc, gm = gm, tighttol = tighttol
#         function cvx_callback(cb_data)
#             varvals = [JuMP.callback_value(cb_data, var) for var in bbc.vars]
#             eval!(bbc, DataFrame(varvals, string.(bbc.vars)))
#             if bbc.Y[end] <= -tighttol
#                 update_gradients(bbc, [size(bbr.X,1)])
#                 con = @build_constraint(0 >= bbc.Y[end]  - 
#                                          sum(Array(bbc.gradients[end,:]) .* (bbc.vars .- varvals)))
#                 MOI.submit(gm.model, MOI.UserCut(cb_data), con)
#             end
#         end
#     end
#     MOI.set(gm.model, MOI.UserCutCallback(), cvx_cb)
# end

# function add_convex_callback(gm::GlobalModel, bbr::BlackBoxRegressor, tighttol = get_param(gm, :tighttol))
#     cvx_cb = let bbr = bbr, gm = gm, tighttol = tighttol
#         function cvx_callback(cb_data)
#             varvals = [JuMP.callback_value(cb_data, var) for var in bbr.vars]
#             dep_var = JuMP.callback_value(bbr.dependent_var)
#             eval!(bbr, DataFrame(varvals, string.(bbr.vars)))
#             if bbr.Y[end] <= dep_var - tighttol
#                 update_gradients(bbr, [size(bbr.X,1)])
#                 con = @build_constraint(bbr.dependent_var >= bbr.Y[end]  - 
#                                          sum(Array(bbr.gradients[end,:]) .* (bbr.vars .- varvals)))
#                 MOI.submit(gm.model, MOI.UserCut(cb_data), con)
#             end
#         end
#     end
#     MOI.set(gm.model, MOI.UserCutCallback(), cvx_cb)
# end

function gm_callback(gm::GlobalModel)
    sol_leaves = find_leaf_of_soln.(gm.bbls)
    var_vals = solution(gm)
for i=1:length(gm.bbls)
    if gm.bbls[i] isa BlackBoxClassifier && gm.feas_history[end][i] != 0 && !gm.bbls[i].equality
        bbl = gm.bbls[i]
        rel_vals = var_vals[:, string.(bbl.vars)]
        eval!(bbl, rel_vals)
        Y = bbl.Y[end]
        update_gradients(bbl, [size(bbl.X, 1)])
        cut_grad = bbl.gradients[end, :]
        push!(bbl.mi_constraints[sol_leaves[i]], 
            @constraint(gm.model, sum(Array(cut_grad) .* (bbl.vars .- Array(rel_vals))) + Y >= 0)) 
    end
    # TODO: add infeasibility cuts for equalities as well. 
end

gm = nlp1(true)
uniform_sample_and_eval!(gm)
learn_constraint!(gm)
add_tree_constraints!(gm)
optimize!(gm)
bbrs = [bbl in gm.bbls if bbl isa BlackBoxRegressor]
bbls = [bbl in gm.bbls if bbl isa BlackBoxClassifier]

update_vexity(gm.bbls[1])

    # cvx_cb = let bbr = bbr, gm = gm, tighttol = tighttol
        function cvx_cb(cb_data)
            status = JuMP.callback_node_status(cb_data, gm.model)
            # if status == MOI.CALLBACK_NODE_STATUS_FRACTIONAL
            #     return
            # elseif status == MOI.CALLBACK_NODE_STATUS_INTEGER
            # else
            #     @assert status == MOI.CALLBACK_NODE_STATUS_UNKNOWN
            # end
            varvals = [JuMP.callback_value(cb_data, var) for var in bbr.vars]
            # varvals = JuMP.getvalue.(bbr.vars)
            dep_var = JuMP.callback_value(bbr.dependent_var)
            # dep_var = JuMP.getvalue(bbr.dependent_var)
            dat = DataFrame(string.(bbr.vars) .=> varvals)
            val = bbr(dat)[1]
            if dep_var <= 11. #val - tighttol
                grad = Array(evaluate_gradient(bbr, dat))
                con = @build_constraint(bbr.dependent_var >= bbr.Y[end]  + 
                                         sum(grad .* (bbr.vars .- varvals)))
                MOI.submit(gm.model, MOI.LazyConstraint(cb_data), con)
            end
            return
        end
    # end
    MOI.set(gm.model, MOI.RawParameter("LazyConstraints"), 1)
    MOI.set(gm.model, MOI.LazyConstraintCallback(), cvx_cb)
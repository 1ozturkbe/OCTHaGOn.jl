# """
# Creates a callback function for the GlobalModel.
# """
function gm_user_cuts(gm::GlobalModel)
    M=1e5
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier & !bbl.equality
            varvals = DataFrame(JuMP.callback_value(cb_data, bbl.vars), string.(bbl.vars))
            eval!(bbl, val_vals)
            if bbl.Y[end] <= -1*get_param(gm, :tighttol)
                update_gradients(bbl, [size(bbl.X, 1)])
                leaf = IAI.apply(bbl.learners[end], varvals)
                leaf_var = bbl.leaf_variables[leaf[1]]
                con = JuMP.build_constraint(sum(Array(bbl.gradients[end]) .* (bbl.vars .- Array(var_vals)))  + 
                                            bbl.Y[end] + M * (1 - leaf_var) >= 0)
                push!(bbl.mi_constraints[leaf[1]], con)
                MOI.submit(gm.model, MOI.UserCut(cb_data), con)
            end
        end
    end
    return
end

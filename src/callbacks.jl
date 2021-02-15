function add_convex_callback(gm::GlobalModel, bbc::BlackBoxClassifier, tighttol = get_param(gm, :tighttol))
    cvx_cb = let bbc = bbc, gm = gm
        function cvx_callback(cb_data)
            varvals = [JuMP.callback_value(cb_data, var) for var in bbc.vars]
            eval!(bbc, DataFrame(varvals, string.(bbc.vars)))
            if bbc.Y[end] <= -tighttol
                update_gradients(bbc, [size(bbr.X,1)])
                con = @build_constraint(0 >= bbc.Y[end]  - 
                                         sum(Array(bbc.gradients[end,:]) .* (bbc.vars .- varvals)))
                MOI.submit(gm.model, MOI.LazyConstraint(cb_data), con)
            end
        end
    end
    MOI.set(gm.model, MOI.LazyConstraintCallback(), cvx_cb)
end

function add_convex_callback(gm::GlobalModel, bbr::BlackBoxRegressor, tighttol = get_param(gm, :tighttol))
    cvx_cb = let bbr = bbr, gm = gm
        function cvx_callback(cb_data)
            varvals = [JuMP.callback_value(cb_data, var) for var in bbr.vars]
            dep_var = JuMP.callback_value(bbr.dependent_var)
            eval!(bbr, DataFrame(varvals, string.(bbr.vars)))
            if bbr.Y[end] <= dep_var - tighttol
                update_gradients(bbr, [size(bbr.X,1)])
                con = @build_constraint(bbr.dependent_var >= bbr.Y[end]  - 
                                         sum(Array(bbr.gradients[end,:]) .* (bbr.vars .- varvals)))
                MOI.submit(gm.model, MOI.LazyConstraint(cb_data), con)
            end
        end
    end
    MOI.set(gm.model, MOI.LazyConstraintCallback(), cvx_cb)
end
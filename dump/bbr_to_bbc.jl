""" Conversion pipe dreams... """
function bbr_to_bbc!(gm::GlobalModel, idx::Array{Int64} = collect(1:length(gm.bbls)))
    for i in idx
        if gm.bbls[i] isa BlackBoxRegressor
            bbr = gm.bbls[i]
            if bbr.constraint isa JuMP.ConstraintRef
                throw(OCTException("Cannot convert BBR $(bbr.name) to a BBC " *
                    "since the bbr.constraint isa JuMP.ConstraintRef."))
            end
            if !isempty(bbr.mi_constraints)
                clear_tree_constraints!(gm, bbr)
            end
            clear_data!(bbr)
            new_bbc = BlackBoxClassifier(constraint = bbr.constraint, 
                                         vars = [bbr.vars..., bbr.dependent_var],
                                         expr_vars = [bbr.expr_vars..., bbr.dependent_var],
                                         equality = bbr.equality, name = bbr.name)
            new_bbc.f(new_bbc.expr_vars...) = bbr.f()
            f = functionify(expr)
            gradable_fn = x -> Base.invokelatest(new_bbc.f, [x[i] for i in var_ranges]...)
            return x -> ForwardDiff.gradient(gradable_fn, x)
            
            if get_param(bbr, :gradients)
                var_ranges = get_var_ranges(new_bbc.expr_vars)
            f = functionify(expr)
            gradable_fn = x -> Base.invokelatest(new_bbc.f, [x[i] for i in var_ranges]...)
            return x -> ForwardDiff.gradient(gradable_fn, x)

            expr_vars = vars
            if isnothing(dependent_var)
                new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                                equality = equality, name = name)
                add_data!(new_bbl, X, Y)
                push!(gm.bbls, new_bbl)
                return
            end
        end
    end
end
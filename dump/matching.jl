
""" Matches bbls to associated variables (except for dependent variables in Regressors). """
function match_bbls_to_vars(bbls::Array,
                            vars::Array = JuMP.all_variables(bbls))
    # TODO: improve types.
    bbl_to_var = Dict()
    for bbl in bbls
        unb = false
        for var in vars
            if var in bbl.vars
                if unb
                    push!(bbl_to_var[bbl], var)
                else
                    bbl_to_var[bbl] = [var]
                    unb = true
                end
            end
        end
    end
    # Sorting so that we penalize both in the number of unbounded and total variables.
    bbl_to_var = sort(collect(bbl_to_var), by= x->length(x[2])*length(x[1].vars)^0.5)
    return bbl_to_var
end

match_bbls_to_vars(gm::GlobalModel, vars::Array = JuMP.all_variables(gm)) = match_bbls_to_vars(gm.bbls, vars)
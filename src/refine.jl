""" Finds the linear min/max bounds of JuMP.VariableRefs."""
function find_linear_bounds!(gm::GlobalModel; bbls::Array{BlackBoxClassifier, BlackBoxRegressor} = gm.bbls, M=1e5, all_bounds::Bool = false)
    unbounds = get_unbounds(bbls)
    if all_bounds
        unbounds = get_bounds(bbls)
    end
    if isnothing(unbounds)
        return
    end
    clear_tree_constraints!(gm)
    m = gm.model
    orig_objective = JuMP.objective_function(m)
    new_bounds = copy(unbounds)
    @showprogress 0.5 "Finding bounds..." for var in collect(keys(unbounds))
        if isinf(unbounds[var][1]) || all_bounds
            @objective(m, Min, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][1] = getvalue(var);
            end
        end
        if isinf(unbounds[var][2]) || all_bounds
            @objective(m, Max, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][2] = getvalue(var);
            end
        end
    end
    # Revert objective, bounds and return new_bounds
    @objective(m, Min, orig_objective)
    bound!(gm, new_bounds)
    return new_bounds
end

"""
    find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxClassifier, BlackBoxRegressor} = [], M = 1e5, all_bounds::Bool=true)

Finds the outer variable bounds of GlobalModel by solving only over the linear constraints
and listed bbls.
TODO: improve! Only find bounds of non-binary variables.
"""

function find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxClassifier, BlackBoxRegressor} = gm.bbls, M = 1e5, all_bounds::Bool=false)
    linear_bounds = find_linear_bounds!(gm, bbls = bbls, M = M, all_bounds = all_bounds)
    return linear_bounds
end
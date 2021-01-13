""" Finds the linear min/max bounds of JuMP.VariableRefs."""
function find_linear_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = gm.bbls, M=1e5, all_bounds::Bool = false)
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
    find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = [], M = 1e5, all_bounds::Bool = false)

Finds the outer variable bounds of GlobalModel by solving only over the linear constraints
and listed bbls.
TODO: improve! Only find bounds of non-binary variables.
"""
function find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = gm.bbls, M = 1e5, all_bounds::Bool = false)
    linear_bounds = find_linear_bounds!(gm, bbls = bbls, M = M, all_bounds = all_bounds)
    return linear_bounds
end

""" 
    ridge_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT, rho::Float64 = 0., weights = ones(length(Y)))

Performs ridge regression on data. 
"""
function ridge_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT, rho::Float64 = 0., weights = ones(length(Y)))
    m = JuMP.Model(with_optimizer(solver))
    normalized_X, lbs, ubs = normalized_data(X);
    @variable(m, x[1:size(X,2)])
    @variable(m, offset)
    @objective(m, Min, sum((Y - normalized_X*x .- offset).^2) + rho*sum(x.^2))
    status = optimize!(m)
    return getvalue(offset), getvalue.(x)./(ubs-lbs)
end

""" Finds upper/lower regressors of data that are conservative. """
function ul_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, α[1:size(X, 2)])
    @variable(m, α0)
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    # function callbackfn(cb_data)
    #     b_max = callback_value(cb_data, β_max)
    #     b0_max = callback_value(cb_data, β0_max)        
    #     b_min = callback_value(cb_data, β_min)
    #     b0_min = callback_value(cb_data, β0_min)
    #     for i=1:length(Y)
    #         if Matrix(X[i,:])*b_max + b0_max <= Y[i]
    #             con = @build_constraint(Matrix(X[i,:])*b_max + b0_max >= Y(i))
    #             MOI.submit(m, MOI.LazyConstraint(cb_data), con)
    #         elseif Matrix(X[i,:])*b_min + b0_min >= Y[i]
    #             con = @build_constraint(Matrix(X[i,:])*b_min + b0_min <= Y(i))
    #             MOI.submit(m, MOI.LazyConstraint(cb_data), con)
    #         end  
    #     end
    # end
    # MOI.set(m, MOI.LazyConstraintCallback(), callbackfn)
    # randoms = 1:length(Y)
    # if length(Y) > 30
    #     randoms = randcycle(MersenneTwister(1234), length(Y))[1:20]
    # end
    # MOI.set(m, MOI.LazyConstraintCallback(), my_callback_function)
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* α) + α0 >= Y[i])
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* β) + β0 <= Y[i])
    @objective(m, Min, sum(((Matrix(X) * α .+ α0) - (Matrix(X) * β .+ β0)).^2))
    optimize!(m)
    return [getvalue(α0), getvalue.(α)], [getvalue(β0), getvalue.(β)]
end

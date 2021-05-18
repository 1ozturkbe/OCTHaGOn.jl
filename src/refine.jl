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

""" 
    u_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)

Finds upper regressors of data that are conservative. 
"""
function u_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)
    if size(X, 1) < 2*size(X, 2)
        @warn("Upper regression doesn't have enough data, thus returning constant bounds. ")
        return maximum(Y), zeros(size(X,2))
    end
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, α[1:size(X, 2)])
    @variable(m, α0)
    @variable(m, err[1:length(Y)] >= 0)
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* α) + α0 >= Y[i])
    @constraint(m, err .>= (Matrix(X) * α .+ α0 - Y))
    @constraint(m, err .>= -(Matrix(X) * α .+ α0 - Y))
    @objective(m, Min, sum(err.^2))
    try
        optimize!(m)
        return getvalue(α0), getvalue.(α)
    catch
        @warn("Infeasible u_regress, returning constant bounds.")
        return maximum(Y), zeros(size(X,2))
    end
end

""" 
    l_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)

Finds lower regressors of data that are conservative. 
"""
function l_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)
    if size(X, 1) < 2*size(X, 2)
        @warn("Lower regression doesn't have enough data, thus returning constant bounds. ")
        return minimum(Y), zeros(size(X,2))
    end
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    @variable(m, err[1:length(Y)] >= 0)
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* β) + β0 <= Y[i])
    @constraint(m, err .>= (Matrix(X) * β .+ β0 - Y))
    @constraint(m, err .>= -(Matrix(X) * β .+ β0 - Y))
    @objective(m, Min, sum(err.^2))
    try
        optimize!(m)
        return getvalue(β0), getvalue.(β)
    catch 
        @warn("Infeasible l_regress, returning constant bounds.")
        return minimum(Y), zeros(size(X,2))
    end
end

""" 
    svm(X::DataFrame, Y::Array, threshold = 0; solver = CPLEX_SILENT)

Finds the unregularized SVM split, where threshold is the allowable error. 
"""
function svm(X::Matrix, Y::Array, threshold = 0; solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, error[1:length(Y)] >= 0)
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    for i=1:length(Y)
        @constraint(m, threshold + error[i] >= Y[i] - β0 - sum(X[i,:] .* β))
        @constraint(m, threshold + error[i] >= -Y[i] + β0 + sum(X[i,:] .* β))
    end
    @objective(m, Min, sum(error))
    optimize!(m)
    return getvalue(β0), getvalue.(β)
end

""" 
    reweight(X::Matrix, mag::Float64 = 10)

Gaussian reweighting of existing data by proximity to previous solution.
Note: mag -> Inf results in uniform reweighting. 
Returns:
- weights: weights of X rows, by Euclidian distance
"""
function reweight(X::Matrix, mag::Float64 = 10.)

    n_samples, n_features = size(X);
    mean = [sum(X[:,i])/n_samples for i=1:n_features];
    std = [sum((X[:,i]-ones(n_samples)* mean[i]).^2)/n_samples for i=1:n_features];
    distance = [sum((X[i,:] - mean).^2 ./std) for i=1:n_samples];
    weights = exp.(-1/mag*distance);
    return weights
end

"""
    add_infeasibility_cuts(gm::GlobalModel)

Adds cuts reducing infeasibility of BBC inequalities. 
"""
function add_infeasibility_cuts!(gm::GlobalModel, M = 1e5)
    #TODO: CHECK REGRESSION EQUALITIES.
    #TODO: ADD CUTS TO LINKEDCONSTRAINTS. 
    var_vals = solution(gm)
    for i=1:length(gm.bbls)
        # Inequality Classifiers
        if get_param(gm.bbls[i], :gradients) && gm.bbls[i] isa BlackBoxClassifier && !gm.bbls[i].equality
            bbc = gm.bbls[i]
            if bbc.feas_gap[end] <= 0
                active_leaf = bbc.active_leaves[1]
                @assert length(bbc.active_leaves) == 1
                rel_vals = var_vals[:, string.(bbc.vars)]
                eval!(bbc, rel_vals)
                Y = bbc.Y[end]
                update_gradients(bbc, [size(bbc.X, 1)])
                cut_grad = bbc.gradients[end, :]
                push!(bbc.mi_constraints[bbc.active_leaves[1]], 
                    @constraint(gm.model, sum(Array(cut_grad) .* (bbc.vars .- Array(rel_vals)')) + Y + 
                                        M*(1 - bbc.leaf_variables[bbc.active_leaves[1]]) >= 0))
            end
            for ll in bbc.lls
                if ll.feas_gap[end] <= 0
                    @assert length(ll.active_leaves) == 1
                    rel_vals = DataFrame(Array(var_vals[:, string.(ll.vars)]), string.(bbc.vars))
                    eval!(bbc, rel_vals)
                    Y = bbc.Y[end]
                    update_gradients(bbc, [size(bbc.X, 1)])
                    cut_grad = bbc.gradients[end, :]
                    push!(ll.mi_constraints[ll.active_leaves[1]], 
                    @constraint(gm.model, sum(Array(cut_grad) .* (ll.vars .- Array(rel_vals)')) + Y + 
                                        M*(1 - ll.leaf_variables[ll.active_leaves[1]]) >= 0))
                end
            end
        # Convex Regressors
        elseif get_param(gm.bbls[i], :gradients) && gm.bbls[i] isa BlackBoxRegressor && gm.bbls[i].convex
            bbr = gm.bbls[i]
            if bbr.feas_gap[end] <= 0
                rel_vals = var_vals[:, string.(bbr.vars)]
                eval!(bbr, rel_vals)
                Y = bbr.Y[end]
                update_gradients(bbr, [size(bbr.X, 1)])
                cut_grad = bbr.gradients[end, :]
                push!(bbr.mi_constraints[1], 
                    @constraint(gm.model, bbr.dependent_var >= sum(Array(cut_grad) .* (bbr.vars .- Array(rel_vals)')) + Y )) 
            end
            for ll in bbr.lls
                if ll.feas_gap[end] <= 0
                    rel_vals = DataFrame(Array(var_vals[:, string.(ll.vars)]), string.(bbr.vars))
                    eval!(bbr, rel_vals)
                    Y = bbr.Y[end]
                    update_gradients(bbr, [size(bbr.X, 1)])
                    cut_grad = bbr.gradients[end, :]
                    push!(ll.mi_constraints[1], 
                        @constraint(gm.model, ll.dependent_var >= sum(Array(cut_grad) .* (ll.vars .- Array(rel_vals)')) + Y)) 
                end
            end
        end
        # TODO: add infeasibility cuts for Regressors that are locally convex. 
        # TODO: add infeasibility cuts for equalities as well. 
    end
    return 
end

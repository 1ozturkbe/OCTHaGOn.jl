""" Very coarse solution for gm to check for feasibility and an initial starting point. """
function surveysolve(gm::GlobalModel)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    if !isempty(bbcs)
        for bbc in bbcs
            learn_constraint!(bbc)
            update_tree_constraints!(gm, bbc)
        end
    end
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        for bbr in bbrs
            learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6, minbucket = 2*length(bbr.vars))
            update_tree_constraints!(gm, bbr)
        end
    end
    optimize!(gm)
    return
end

"""
    function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    function add_relaxation_variables!(gm::GlobalModel, bbls::Array)

Populates relax_var attributes of all substructs. 
"""
function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa Real
        bbl.relax_var = @variable(gm.model)
        @constraint(gm.model, bbl.relax_var >= 0)  
    end
    if !isempty(bbl.mi_constraints)
        clear_tree_constraints!(gm, bbl)
        @info "Clearing MI constraints from $(bbl.name) due to relaxation."
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            add_relaxation_variables!(gm, ll)
        end
    end
end

function add_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        add_relaxation_variables!(gm, bbl)
    end
end

add_relaxation_variables!(gm::GlobalModel) = add_relaxation_variables!(gm, gm.bbls)

function clear_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa JuMP.VariableRef
        is_valid(gm.model, bbl.relax_var) || 
            throw(OCTException("$(bbl.relax_var) is not a valid variable of  $(gm.model)."))
        delete(gm.model, bbl.relax_var)
        bbl.relax_var = 0
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            clear_relaxation_variables!(gm, ll)
        end
    end
end

function clear_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        clear_relaxation_variables!(gm, bbl)
    end
end

clear_relaxation_variables!(gm::GlobalModel) = clear_relaxation_variables!(gm, gm.bbls)

"""
Includes relaxation variables in objective function 
"""
function relaxed_objective!(gm::GlobalModel, M::Real = 1e8)
    no = 0
    for bbl in gm.bbls
        no += M*bbl.relax_var
        for ll in bbl.lls
            no += M*ll.relax_var
        end
    end
    @objective(gm.model, Min, gm.objective + no)
    return
end

""" 
Removes relaxation variables from objective. 
"""
function tight_objective!(gm::GlobalModel)
    @objective(gm.model, Min, gm.objective)
end

"""
    descend!(gm::GlobalModel; 
            max_iterations = 100, step_penalty = 1e4, step_size = 1e-3, decay_rate = 2)

Performs gradient descent on the last optimal solution in gm.solution_history.
In case of infeasibility, first projects the feasible point using the local
constraint gradients. 

# Optional arguments:
max_iterations: maximum number of gradient steps or projections.
step_penalty: magnitude of penalty on step size during projections.
step_size: Size of 0-1 normalized Euclidian ball we can step. 
decay_rate: Exponential coefficient of step size reduction. 

"""
function descend!(gm::GlobalModel; 
                 max_iterations = 100, step_penalty = 1e4, step_size = 1e-3, decay_rate = 2)
    clear_tree_constraints!(gm)

    # Initialization
    bbls = gm.bbls
    vars = gm.vars
    gm_bounds = get_bounds(vars)

    # Checking for a nonlinear objective
    obj_bbl = nothing
    if gm.objective isa VariableRef
        obj_bbl = filter(x -> x.dependent_var == gm.objective, 
                            [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])
        if isempty(obj_bbl)
            obj_bbl = nothing
        elseif length(obj_bbl) == 1
            obj_bbl = obj_bbl[1]
            vars = [var for var in vars if var != obj_bbl.dependent_var]
            bbls = [bbl for bbl in gm.bbls if bbl != obj_bbl]
        elseif length(obj_bbl) > 1
            throw(OCTException("GlobaModel $(gm.name) has more than one BlackBoxRegressor" *
                               " on the objective variable."))
        end
    end 
    
    var_max = [maximum(gm_bounds[key]) for key in vars]
    var_min = [minimum(gm_bounds[key]) for key in vars]
    grad_shell = DataFrame([Float64 for i=1:length(vars)], string.(vars))

    # Final checks
    obj_gradient = copy(grad_shell)
    if isnothing(obj_bbl)
        if gm.objective isa VariableRef
            append!(obj_gradient, DataFrame(string.(gm.objective) => 1), cols = :subset)
        elseif gm.objective isa JuMP.GenericAffExpr
            append!(obj_gradient, DataFrame(Dict(string(key) => value for (key,value) in gm.objective.terms)), cols = :subset)
        end
        obj_gradient = coalesce.(obj_gradient, 0)
    end

    # Last solution recheck, and actual objective computation
    x0 = DataFrame(gm.solution_history[end, :])
    if !isnothing(obj_bbl)
        x0[:, string(gm.objective)] = obj_bbl(x0)
    end
    sol_vals = x0[:,string.(vars)]
    feas_gap(gm, x0) # Checking feasibility gaps
    append!(gm.solution_history, x0) # Objective "projection"

    # Descent direction, counting and book-keeping
    d = @variable(gm.model, [1:length(vars)])
    ct = 0
    d_improv = 1e5
    abstol = get_param(gm, :abstol)

    # WHILE LOOP
    @info("Starting projected gradient descent...")
    while ((!all([bbl.feas_gap[end] for bbl in gm.bbls] .>= 0) ||
         abs(d_improv) >= get_param(gm, :abstol)) && ct < max_iterations) || ct == 0
        prev_obj = gm.cost[end]
        constrs = []
        ct += 1
    
        # Linear objective gradient and constraints
        if !isnothing(obj_bbl)
            update_gradients(obj_bbl, [length(obj_bbl.Y)])
            obj_gradient = copy(grad_shell)
            append!(obj_gradient, 
                DataFrame(obj_bbl.gradients[end,:]), cols = :subset) 
            obj_gradient = coalesce.(obj_gradient, 0)
            # Update objective constraints
            append!(constrs, [@constraint(gm.model, sum(Array(obj_gradient[end,:]) .* d) + 
                              obj_bbl.dependent_var >= obj_bbl.Y[end])])
        end

        # Initializing descent direction
        push!(constrs, @constraint(gm.model, d .== vars .- Array(sol_vals[end,:])))
        if all([bbl.feas_gap[end] for bbl in gm.bbls] .>= 0)
            push!(constrs, @constraint(gm.model, sum((d ./ (var_max .- var_min)).^2) <= 
                    step_size/exp(decay_rate*(ct-1)/max_iterations)))
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d))
        else # Project if the current solution is infeasible. 
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d) + 1e4*sum((d ./ (var_max .- var_min)).^2))
        end

        # Constraint evaluation
        for bbl in bbls
            update_gradients(bbl, [length(bbl.Y)])
            constr_gradient = copy(grad_shell)
            append!(constr_gradient, DataFrame(bbl.gradients[end,:]), cols = :subset)
            constr_gradient = coalesce.(constr_gradient, 0)
            if bbl isa BlackBoxClassifier # TODO: add LL cuts
                if bbl.equality
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] == 0))
                else
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + bbl.Y[end] >= 0))
                end
            elseif bbl isa BlackBoxRegressor
                if bbl.equality
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var == bbl.Y[end]))
                else
                    push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + bbl.dependent_var >= bbl.Y[end]))
                end
            end
            if !isempty(bbl.lls)
                n_lls = length(bbl.lls)
                update_gradients(bbl, collect(length(bbl.Y) - n_lls:length(bbl.Y)-1))
                for i = 1:n_lls
                    ll = bbl.lls[i]
                    constr_gradient = copy(grad_shell)
                    append!(constr_gradient, # Changing the headers of the gradient 
                        DataFrame(string.(ll.vars) .=> Array(bbl.gradients[end-n_lls-1+i,:])), cols = :subset)
                    constr_gradient = coalesce.(constr_gradient, 0)
                    Y_val = bbl.Y[end-n_lls-1+i]
                    if bbl isa BlackBoxClassifier # TODO: add LL cuts
                        if bbl.equality
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val == 0))
                        else
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d)  + Y_val >= 0))
                        end
                    elseif bbl isa BlackBoxRegressor
                        if bbl.equality
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var == Y_val))
                        else
                            push!(constrs, @constraint(gm.model, sum(Array(constr_gradient[end,:]) .* d) + ll.dependent_var >= Y_val))
                        end
                    end
                end
            end
        end

        # Optimizing the step, and finding next x0
        optimize!(gm.model)
        # @warn "Last solution was infeasible. Please check, reinitialize x0 with new trees," *
        #         " or perhaps initialize with another x0."
        # TODO: perhaps try another method to restore feasibility? 
        x0 = DataFrame(string.(vars) .=> getvalue.(vars))
        if !isnothing(obj_bbl)
            x0[!, string(gm.objective)] = obj_bbl(x0)
        end
        sol_vals = x0[:,string.(vars)]
        feas_gap(gm, x0)
        append!(gm.solution_history, x0)

        # Measuring improvements
        if !isnothing(obj_bbl)
            push!(gm.cost, gm.solution_history[end, string(gm.objective)])
        else
            push!(gm.cost,  JuMP.getvalue(gm.objective))
        end
        d_improv = gm.cost[end-1] - gm.cost[end]

        # Delete gradient constraints (TODO: perhaps add constraints to avoid cycling?)
        for con in constrs
            delete(gm.model, con)
        end
    end

    if ct >= max_iterations && abs(d_improv) >= abstol
        @info("Max iterations reached, but descent not converged to tolerance!" * 
              " Please descend further, perhaps with reduced step sizes.")
    elseif ct >= max_iterations && !all([bbl.feas_gap[end] for bbl in gm.bbls] .>= 0)
        @info("Max iterations reached, but solution is not feasible!" * 
              " Please observe the cost evolution, descend again, or relax your constraints.")
    else
        @info("PGD converged in $(ct) iterations!")
    end

    # Reverting objective, and deleting vars
    @objective(gm.model, Min, gm.objective)
    delete(gm.model, d)

    # Returning final solution
    return gm.solution_history[end,:]
end
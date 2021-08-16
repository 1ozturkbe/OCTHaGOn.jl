function test_baron_solve(m::JuMP.Model = gear(false))
    set_optimizer(m, BARON_SILENT)
    optimize!(m)
    sol = solution(m)
    @test true
end

function test_speed_params(gm::GlobalModel = minlp(true), solver = CPLEX_SILENT)
    set_optimizer(gm, solver)   
    bbl = gm.bbls[1]
    uniform_sample_and_eval!(bbl)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [3, 5]
    tree_mat = [[], []]
    time_mat = [[], []]
    for i=1:length(ls_num_hyper_restarts)
        for j=1:length(ls_num_tree_restarts)
            t1 = time()
            params = Dict(:ls_num_hyper_restarts => ls_num_hyper_restarts[i],
                          :ls_num_tree_restarts => ls_num_tree_restarts[j],
                          :max_depth => 2)
            learn_constraint!(bbl; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbl.learners[end])
        end
    end
    @test true
end

function test_classify_gradients()
    gm = minlp(true)
    bbr = gm.bbls[3]
    uniform_sample_and_eval!(gm)
    idxs = collect(1:10)
    classify_curvature(bbr, idxs)
    @test !any(ismissing.(bbr.curvatures[idxs]))
    classify_curvature(bbr)
    @test all(bbr.curvatures .> 0)
    update_vexity(bbr)
    @test bbr.convex == true
    @test bbr.local_convexity == 1.
end

function test_infeasibility_cuts()
    gm = sagemark_to_GlobalModel(15, false)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    add_tree_constraints!(gm)
    optimize!(gm)
    bbc_idxs = [bbl isa BlackBoxClassifier for bbl in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_feasibility_sample()
    gm = speed_reducer()
    uniform_sample_and_eval!(gm)
    [set_param(bbl, :threshold_feasibility, 0.3) for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    @test any(check_feasibility(gm) .!= 1)
    feasibility_sample(gm)
    @test all(check_feasibility(gm) .== 1)
end

function test_survey_method(gm::GlobalModel = minlp(true))
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    surveysolve(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    bbc_idxs = [x isa BlackBoxClassifier for x in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_concave_regressors(gm::GlobalModel = gear(true))
    gm = gear(true)
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        update_vexity.(bbrs)  
    end
    bbr = bbrs[1]
    
    # Checking number of constraints
    types = JuMP.list_of_constraint_types(gm.model)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    surveysolve(gm) # 1st tree (ORT)
    actual = bbr.actuals[end]
    optim = bbr.optima[end]
    learn_constraint!(bbr, "upper" => minimum(bbr.actuals)) # 2nd tree (Upper OCT)
    learn_constraint!(bbr, "reg" => 40, 
                        regression_sparsity = 0, max_depth = 3) # 3th tree (Regressor on upper bounded samples)

    # Trying to add and remove individual constraints in random order to make sure no constraints accidentally remain. 
    for i=1:length(bbr.learners)
        update_tree_constraints!(gm, bbr, i)
        for j=1:length(bbr.learners)
            update_tree_constraints!(gm, bbr, j)
            for k = 1:length(bbr.learners)
                update_tree_constraints!(gm, bbr, k)
                treekeys = collect(keys(bbr.active_trees))
                treevalues = collect(values(bbr.active_trees))
                if length(treevalues) >= 2
                    @test Pair(treevalues[1].first, treevalues[2].first) in OCT.valid_pairs
                elseif length(treevalues) == 1
                    @test treevalues[1].first in OCT.valid_singles || treevalues[1].first == "upper"
                end
                clear_tree_constraints!(gm)
                n_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
                @test n_constraints == init_constraints
            end
        end
    end
    update_tree_constraints!(gm, bbr, 2)
    update_tree_constraints!(gm, bbr, 3)
    @test active_lower_tree(bbr) == 3
    @test active_upper_tree(bbr) == 3  
    optimize!(gm)
    clear_tree_constraints!(gm)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
end

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)
    for bbl in gm.bbls
        learn_constraint!(bbl)
        add_tree_constraints!(gm, bbl)
    end
    optimize!(gm)    
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while (gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
end 

function optimize_and_time!(m::Union{JuMP.Model, GlobalModel})
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        recipe(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
end

# test_baron_solve()

# test_speed_params()

# test_classify_gradients()

# test_infeasibility_cuts()

# test_feasibility_sample()

# test_survey_method()

# test_concave_regressors()

"""
    descend(gm::GlobalModel; 
            max_iterations = 100, step_size = 1e-3, decay_rate = 2)

Performs gradient descent on the last optimal solution in gm.solution_history.
In case of infeasibility, first projects the feasible point using the local
constraint gradients. 

# Optional arguments:
max_iterations: maximum number of gradient steps or projections.
step_size: Size of 0-1 normalized Euclidian ball we can step. 
decay_rate: Exponential coefficient of step size reduction. 

"""
function descend(gm::GlobalModel; 
                 max_iterations = 100, step_size = 1e-3, decay_rate = 2)
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

    # WHILE LOOP
    @info("Starting gradient descent...")
    while !all([bbl.feas_gap[end] for bbl in gm.bbls] .>= 0) ||
        (ct < max_iterations && abs(d_improv) >= get_param(gm, :abstol)) || ct == 0
        constrs = []
        ct += 1
        @info("Count $(ct)")

        # Initializing descent direction
        push!(constrs, @constraint(gm.model, d .== vars .- Array(sol_vals[end,:])))
        if all([bbl.feas_gap[end] for bbl in gm.bbls] .>= 0)
            push!(constrs, @constraint(gm.model, sum((d ./ (var_max .- var_min)).^2) <= 
                    step_size/exp(decay_rate*(ct-1)/max_iterations)))
            @objective(gm.model, Min, sum(Array(obj_gradient[end,:]) .* d))
                # @objective(gm.model, Min, gm.objective)
        else # Project if the current solution is infeasible. 
            @objective(gm.model, Min, gm.objective + JuMP.upper_bound(gm.objective)*sum((d ./ (var_max .- var_min)).^2))
        end

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

        # Delete gradient constraints (TODO: perhaps add constraints to avoid cycling?)
        for con in constrs
            delete(gm.model, con)
        end
        d_improv = gm.solution_history[end-1, string(gm.objective)] - gm.solution_history[end, string(gm.objective)]
    end

    if ct >= max_iterations && abs(d_improv) >= get_param(gm, :abstol)
        @info("Max iterations reached, but not converged! Please descend further, perhaps with reduced step sizes.")
    end

    # Reverting objective, and deleting vars
    @objective(gm.model, Min, gm.objective)
    delete(gm.model, d)

    # Returning final solution
    return gm.solution_history[end,:]
end

# Implementing gradient descent

gm = sagemark_to_GlobalModel(25, false)
# gm = nlp2(true)
# gm = speed_reducer()
# gm = minlp(true)
set_param(gm, :abstol, 1e-4)
set_param(gm, :ignore_accuracy, true)
uniform_sample_and_eval!(gm)
learn_constraint!(gm)
add_tree_constraints!(gm)
set_optimizer(gm, CPLEX_SILENT)
optimize!(gm)
descend(gm)

# Different problems to test different descent aspects
# nlp2 for equalities
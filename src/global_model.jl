"""
Contains all constraints and data to solve the global optimization optimization problem. Add linear constraints to a `GlobalModel` through `JuMP.@constraint(gm.model, ...)`, and nonlinear constraints using `add_nonlinear_constraint`.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                            # Associated JuMP.Model
    name::String = "Model"                                       # Name
    bbls::Array{BlackBoxLearner} = BlackBoxLearner[]             # Constraints to be learned
    vars::Array{JuMP.VariableRef} = JuMP.all_variables(model)    # JuMP variables
    objective = JuMP.objective_function(model)                # Original objective function
    solution_history::DataFrame = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)]) # Solution history
    cost::Array = []                                             # List of costs. 
    soldict::Dict = Dict()                                       # For solution extraction
    params::Dict = gm_defaults()                                 # GM settings
end

function Base.show(io::IO, gm::GlobalModel)
    println(io, "GlobalModel " * gm.name * " with $(length(gm.vars)) variables: ")
    println(io, "Has $(length(gm.bbls)) BlackBoxLearners.")
    if get_param(gm, :ignore_feasibility)
        if get_param(gm, :ignore_accuracy)
            println(io, "Ignores training accuracy and data_feasibility thresholds.")
        else
            println(io, "Ignores data feasibility thresholds.")
        end
    else
        if get_param(gm, :ignore_accuracy)
            println("Ignores training accuracy thresholds.")
        end
    end
end

function set_param(gm::GlobalModel, key::Symbol, val) 
    set_param(gm.params, key, val)
    for bbl in gm.bbls
        set_param(bbl.params, key, val, false)
    end
end

get_param(gm::GlobalModel, key::Symbol) = get_param(gm.params, key)

"""
    (gm::GlobalModel)(name::String)

Finds BlackBoxLearner in GlobalModel by name.
"""
function (gm::GlobalModel)(name::String)
    fn_names = getfield.(gm.bbls, :name)
    fns = gm.bbls[findall(x -> x == name, fn_names)]
    if length(fns) == 1
        return fns[1]
    elseif length(fns) == 0
        @warn("No constraint with name ", name)
        return
    else
        @warn("Multiple constraints with name ", name)
        return fns
    end
end

""" Finds active leaves of all constraints in GlobalModel."""
function active_leaves(gm::GlobalModel)
    for bbl in gm.bbls
        active_leaves(bbl)
        for ll in bbl.lls
            active_leaves(ll)
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Extends JuMP.all_variables to GlobalModels and BlackBoxLearners, but only returns non-auxiliary variables
"""
JuMP.all_variables(bbo::Union{GlobalModel, BlackBoxLearner, LinkedLearner}) = bbo.vars

function JuMP.all_variables(bbls::Union{Array{BlackBoxLearner}, Array{LinkedLearner}})
    return unique(Iterators.flatten(([JuMP.all_variables(bbl) for bbl in bbls])))
end

""" Extends JuMP.set_optimizer to GlobalModels. """
function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    JuMP.set_optimizer(gm.model, optimizer_factory)
end

count_types(gm::GlobalModel) = count_types(gm.model)
count_variables(gm::GlobalModel) = count_variables(gm.model)
count_constraints(gm::GlobalModel) = count_constraints(gm.model)

"""
    $(TYPEDSIGNATURES)

Returns bounds of all variables.
"""
function get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxLearner, 
                                 Array{BlackBoxLearner}, LinkedLearner, 
                                 Array{LinkedLearner}})
    return get_bounds(JuMP.all_variables(model))
end

""" Returns bounds of all variables, flattened in order of .vars attribute. """
function flattened_bounds(bbc::Union{GlobalModel, BlackBoxLearner, LinkedLearner})
    var_bounds = get_bounds(bbc)
    return [var_bounds[var] for var in bbc.vars]
end

"""
    $(TYPEDSIGNATURES)

Returns only unbounded variables. 
"""
function get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxLearner, 
                      Array{BlackBoxLearner}})
    return get_unbounds(JuMP.all_variables(model))
end

"""
    $(TYPEDSIGNATURES)

Takes on parsing and allocation of variables depending on user input. 
"""
function determine_vars(gm::GlobalModel,
                        constraint::Union{JuMP.ConstraintRef, Expr};
                        vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                        expr_vars::Union{Nothing, Array} = nothing)
    if constraint isa JuMP.ConstraintRef
        if isnothing(vars) && isnothing(expr_vars)
            vars = vars_from_constraint(constraint)
            return vars, vars
        elseif !isnothing(vars)
            return vars, vars
        end
    else
        if isnothing(vars) && isnothing(expr_vars)
            expr_vars = vars_from_expr(constraint, gm.model)
            vars = flat(expr_vars)
            return vars, expr_vars
        elseif isnothing(expr_vars)
            expr_vars = vars_from_expr(constraint, gm.model)
            @assert length(flat(expr_vars)) >= length(vars)
            return vars, expr_vars
        elseif isnothing(vars)
            vars = flat(expr_vars)
            return vars, expr_vars
        else
            @assert length(flat(expr_vars)) >= length(vars)
            return vars, expr_vars
        end
    end
end

"""
    add_nonlinear_constraint(gm::GlobalModel,
                     constraint::Union{JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = gm.name * " " * string(length(gm.bbls) + 1),
                     equality::Bool = false)

 Adds a new nonlinear constraint to Global Model. Standard method for adding BlackBoxClassifiers and BlackBoxRegressors. 

 Note: If adding constraints via an `Expr`, please define the appropriate vector indices. Eg. 
    `expr = :((x) -> -x[1:\$(N)]'*\$(Q)*x[1:\$(N)] - \$(c)'*x[1:\$(N)])`,
instead of 
    `expr = :((x) -> -x'*\$(Q)*x - \$(c)'*x)`,
to avoid vectorization errors.
"""
function add_nonlinear_constraint(gm::GlobalModel,
                     constraint::Union{JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = "bbl" * string(length(gm.bbls) + 1),
                     equality::Bool = false)
    vars, expr_vars = determine_vars(gm, constraint, vars = vars, expr_vars = expr_vars)
    if constraint isa Expr
        if isnothing(dependent_var)
            new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                         equality = equality, name = name)
            set_param(new_bbl, :n_samples, Int(ceil(get_param(gm, :sample_coeff)*sqrt(length(vars)))))
            push!(gm.bbls, new_bbl)
            return
        else
            new_bbl = BlackBoxRegressor(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        dependent_var = dependent_var, equality = equality, name = name)
            set_param(new_bbl, :n_samples, Int(ceil(get_param(gm, :sample_coeff)*sqrt(length(vars)))))
            push!(gm.bbls, new_bbl)
            return
        end
    elseif constraint isa JuMP.ConstraintRef
        !isnothing(dependent_var) && throw(OCTHaGOnException("Constraint " * name * " is of type $(string(typeof(constraint))) " *
                                                        "and cannot have a dependent variable " * string(dependent_var) * "."))
        new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        equality = equality, name = name)
        set_param(new_bbl, :n_samples, Int(ceil(get_param(gm, :sample_coeff)*sqrt(length(vars)))))
        push!(gm.bbls, new_bbl)
        JuMP.delete(gm.model, constraint)   
    end
end

"""
    function add_nonlinear_or_compatible(gm::GlobalModel,
        constraint::Union{JuMP.ConstraintRef, Expr};
        vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
        expr_vars::Union{Nothing, Array} = nothing,
        dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
        name::String = gm.name * "_" * string(length(gm.bbls) + 1),
        equality::Bool = false)

Extends `add_nonlinear_constraint` to recognize JuMP compatible linear constraints and add them as linear JuMP constraints instead.
"""
function add_nonlinear_or_compatible(gm::GlobalModel,
                     constraint::Union{JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = gm.name * "_" * string(length(gm.bbls) + 1),
                     equality::Bool = false)
    vars, expr_vars = determine_vars(gm, constraint, vars = vars, expr_vars = expr_vars)
    fn = functionify(constraint)
    if fn isa Function
        try
            constr_expr = Base.invokelatest(fn, expr_vars...)
            if constr_expr isa JuMP.GenericAffExpr || get_param(gm, :convex_constrs)
                if equality && !isnothing(dependent_var)
                    @constraint(gm.model, dependent_var == constr_expr)
                elseif equality
                    @constraint(gm.model, constr_expr == 0)
                elseif !isnothing(dependent_var)
                    @constraint(gm.model, dependent_var >= constr_expr)
                else 
                    @constraint(gm.model, constr_expr >= 0)
                end
            else
                throw(ErrorException())
            end
        catch
            add_nonlinear_constraint(gm, constraint, vars = vars, expr_vars = expr_vars, 
                                     dependent_var = dependent_var,
                                     name = name, equality = equality)
        end
    else
        add_nonlinear_constraint(gm, constraint, vars = vars, expr_vars = expr_vars, 
                                    dependent_var = dependent_var,
                                    name = name, equality = equality)
    end
end

"""
    add_linked_constraint(gm::GlobalModel, bbc::BlackBoxClassifier, linked_vars::Array{JuMP.Variable})
    add_linked_constraint(gm::GlobalModel, bbr::BlackBoxRegressor, linked_vars::Array{JuMP.Variable}, linked_dependent::JuMP.Variable)

Adds a linked constraint of the same structure as the BlackBoxLearner. 
When a nonlinear constraint is repeated more than once, this function allows the underlying
approximator to be replicated without retraining trees for each constraint.  
Note that the bounds used for sampling are for the original variables of the BlackBoxLearner, so be careful!
"""
function add_linked_constraint(gm::GlobalModel, bbc::BlackBoxClassifier, vars::Array{JuMP.VariableRef})
    length(vars) == length(bbc.vars) || throw(OCTHaGOnException("BBC $(bbc.name) does not" *
    " have the same number of variables as linked variables $(vars)."))
    if !isempty(bbc.mi_constraints)
        clear_tree_constraints!(gm, bbc)
        @info "Cleared constraints from BBC $(bbc.name) since it was relinked."
    end
    lc = LinkedClassifier(vars = vars, equality = bbc.equality)
    var_bounds = flattened_bounds(bbc)
    lc_var_bounds = flattened_bounds(lc)
    all(minimum.(var_bounds) .<= minimum.(lc_var_bounds)) &&  all(maximum.(var_bounds) .>= maximum.(lc_var_bounds)) || throw(ErrorException("The LinkedClassifier must have a smaller variable range than the BlackBoxClassifier."))
    push!(bbc.lls, lc)
    return
end

function add_linked_constraint(gm::GlobalModel, bbr::BlackBoxRegressor, vars::Array{JuMP.VariableRef}, 
                                dependent_var::JuMP.VariableRef)
    length(vars) == length(bbr.vars) || throw(OCTHaGOnException("BBR $(bbr.name) does not" *
    " have the same number of variables as linked variables $(vars)"))
    if !isempty(bbr.mi_constraints)
        clear_tree_constraints!(gm, bbr)
        @info "Cleared constraints from BBR $(bbr.name) since it was relinked."
    end
    lr = LinkedRegressor(vars = vars, dependent_var = dependent_var, 
                                   equality = bbr.equality)
    var_bounds = flattened_bounds(bbr)
    lr_var_bounds = flattened_bounds(lr)
    all(minimum.(var_bounds) .<= minimum.(lr_var_bounds)) &&  all(maximum.(var_bounds) .>= maximum.(lr_var_bounds)) || throw(ErrorException("The LinkedRegressor must have a smaller variable range than the BlackBoxRegressor."))
    push!(bbr.lls, lr)
    return
end

"""
    nonlinearize!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    nonlinearize!(gm::GlobalModel)

Turns gm.model into the nonlinear representation of global optimization problem.
NOTE: to get back to MI-compatible forms, must regenerate GlobalModel from scratch.
"""
function nonlinearize!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    for (i, bbl) in enumerate(bbls)
        if bbl.constraint isa JuMP.ConstraintRef
            JuMP.add_constraint(gm.model, bbl.constraint)
        elseif bbl.constraint isa Expr
            symb = Symbol(bbl.name)
            vars = flat(bbl.expr_vars) # We want flattening of dense vars.
            var_ranges = get_var_ranges(bbl.expr_vars)
            expr = bbl.constraint
            flat_expr = :((x...) -> $(expr)([x[i] for i in $(var_ranges)]...))
            fn = eval(flat_expr)
            JuMP.register(gm.model, symb, length(vars), fn; autodiff = true)
            expr = Expr(:call, symb, vars...)
            if bbl.equality && bbl isa BlackBoxRegressor
                JuMP.add_NL_constraint(gm.model, :($(expr) == $(bbl.dependent_var)))
            elseif bbl.equality
                JuMP.add_NL_constraint(gm.model, :($(expr) == 0))
            elseif bbl isa BlackBoxRegressor
                JuMP.add_NL_constraint(gm.model, :($(expr) <= $(bbl.dependent_var)))
            else
                JuMP.add_NL_constraint(gm.model, :($(expr) >= 0))
            end
        end
    end
    return
end

function nonlinearize!(gm::GlobalModel)
    nonlinearize!(gm, gm.bbls)
end

function bound!(model::GlobalModel, bounds::Union{Pair,Dict})
    bound!(model.model, bounds)
end

""" Separates and returns linear and nonlinear constraints in a model. """
function classify_constraints(model::Union{GlobalModel, JuMP.Model})
    jump_model = model
    if model isa GlobalModel
        jump_model = model.model
    end
    all_types = list_of_constraint_types(jump_model)
    nl_constrs = [];
    l_constrs = [];
    l_vartypes = [JuMP.VariableRef, JuMP.GenericAffExpr{Float64, VariableRef}]
    l_constypes = [MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}]
    for (vartype, constype) in all_types
        constrs_of_type = JuMP.all_constraints(jump_model, vartype, constype)
        if any(vartype .== l_vartypes) && any(constype .== l_constypes)
            append!(l_constrs, constrs_of_type)
        else
            append!(nl_constrs, constrs_of_type)
        end
    end
    if !isnothing(jump_model.nlp_data)
        append!(nl_constrs, jump_model.nlp_data.nlconstr)
    end
    return l_constrs, nl_constrs
end

""" Returns the feasibility of data points in a bbl or GM. """
feasibility(bbc::BlackBoxClassifier) = bbc.feas_ratio
feasibility(bbr::BlackBoxRegressor) = size(bbr.X, 1) / (size(bbr.X, 1) + size(bbr.infeas_X, 1))
feasibility(bbls::BlackBoxLearner) = feasibility.(bbls)
feasibility(gm::GlobalModel) = feasibility.(gm.bbls)

""" Returns the accuracy of learners in a bbl or GM. """
function evaluate_accuracy(bbc::BlackBoxClassifier)
    if bbc.convex
        return 1. 
    elseif bbc.feas_ratio in [1., 0]
        @warn(string("Accuracy of BlackBoxClassifier ", bbc.name, " is tautological."))
        return 1.
    elseif isempty(bbc.learners)
        throw(OCTHaGOnException(string("BlackBoxClassifier ", bbc.name, " has not been trained yet.")))
    else
        return bbc.accuracies[end]
    end
end

function evaluate_accuracy(bbr::BlackBoxRegressor)
    if bbr.convex && !bbr.equality
        return 1.
    elseif isempty(bbr.learners)
        throw(OCTHaGOnException(string("BlackBoxRegressor ", bbr.name, " has not been trained yet.")))
    else
        return bbr.accuracies[active_lower_tree(bbr)] 
    end
end

evaluate_accuracy(gm::GlobalModel) = evaluate_accuracy.(gm.bbls)

""" 
    JuMP.optimize!(gm::GlobalModel; kwargs...)

Applies JuMP.optimize! to GlobalModels, and saves solution history. 
"""
function JuMP.optimize!(gm::GlobalModel; kwargs...)
    JuMP.optimize!(gm.model, kwargs...) # Solves the optimization problem. 
    append!(gm.solution_history, solution(gm), cols=:intersect) # Saves the solution. 
    feas_gap(gm) # Computes the feasibility gaps of all constraints.
    push!(gm.cost, JuMP.getobjectivevalue(gm.model)) # Updates the final cost.
    active_leaves(gm) # Updates the active leaves of all approximations. 
    gm.soldict = Dict(key => JuMP.getvalue.(gm.model[key]) for (key, value) in gm.model.obj_dict)
    return
end

"""
    solution(gm::GlobalModel)
    solution(m::JuMP.Model)

Returns the optimal solution of the GlobalModel/JuMP.Model in a DataFrame.
"""
function solution(gm::GlobalModel)
    vals = getvalue.(gm.vars)
    return DataFrame(vals', string.(gm.vars))
end

function solution(m::JuMP.Model)
    variables = JuMP.all_variables(m)
    vals = getvalue.(variables)
    return DataFrame(vals', string.(variables))
end

""" 
    feas_gap(gm::GlobalModel, soln = solution(gm))

Evaluates relative feasibility gap of each nonlinear constraint at the given solution. 
Negative values -> constraint violation for BBCs, 
                    regression underestimation for BBRs. 
Positive values -> constraint violation for BBC equalities, 
                    regression overestimation for BBRs.
"""
function feas_gap(gm::GlobalModel, soln = solution(gm))
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier && !isnothing(bbl.constraint)
            for ll in bbl.lls # LL feas_gaps evaluated first, for descent function
                eval!(bbl, DataFrame(string.(bbl.vars) .=> values(soln[1, string.(ll.vars)])))
                push!(ll.feas_gap, bbl.Y[end] ./ (bbl.max_Y - bbl.min_Y))
            end
            eval!(bbl, soln)
            push!(bbl.feas_gap, bbl.Y[end] ./ (bbl.max_Y - bbl.min_Y))
        elseif bbl isa BlackBoxRegressor && !isnothing(bbl.constraint)
            for ll in bbl.lls # LL feas_gaps evaluated first, for descent function
                eval!(bbl, DataFrame(string.(bbl.vars) .=> values(soln[1, string.(ll.vars)])))
                optimum = soln[:, string(ll.dependent_var)][1]
                actual = bbl.Y[end]
                push!(ll.optima, optimum)
                push!(ll.actuals, actual)
                push!(ll.feas_gap, (optimum-actual) / ((bbl.max_Y - bbl.min_Y)))
            end
            eval!(bbl, soln)
            optimum = soln[:, string(bbl.dependent_var)][1]
            actual = bbl.Y[end]
            push!(bbl.optima, optimum)
            push!(bbl.actuals, actual)
            push!(bbl.feas_gap, (optimum-actual) / ((bbl.max_Y - bbl.min_Y)))
        elseif bbl isa BlackBoxClassifier && isnothing(bbl.constraint)
            for ll in bbl.lls # LL feas_gaps evaluated first, for descent function
                push!(ll.feas_gap, 0)
            end
            push!(bbl.feas_gap, 0) # data constraints are always feasible
        elseif bbl isa BlackBoxRegressor && isnothing(bbl.constraint)
            for ll in bbl.lls # LL feas_gaps evaluated first, for descent function
                optimum = soln[:, string(ll.dependent_var)][1]
                push!(ll.optima, optimum)
                push!(ll.feas_gap, 0) # data constraints are always feasible
            end
            optimum = soln[:, string(bbl.dependent_var)][1]
            push!(bbl.optima, optimum)
            push!(bbl.feas_gap, 0) # data constraints are always feasible
        end
    end
    return
end

""" Prints the last feasibility gap of each constraint. """
function print_feas_gaps(gm::GlobalModel)
    @info "Feasibility gaps:"
    for bbl in gm.bbls
        println("$(bbl.name): $(bbl.feas_gap[end])")
        for ll in bbl.lls
            println("Linked: $(ll.feas_gap[end])")
        end
    end
    return
end

""" Shows feasibility of last solution w.r.t. each approximated constraint. """
function is_feasible(bbl::Union{BlackBoxLearner, LinkedLearner}, tighttol = 1e-8)
    if bbl.equality
        return abs(bbl.feas_gap[end]) <= tighttol
    else
        return bbl.feas_gap[end] >= -tighttol
    end
end

""" Returns the feasibility of the GlobalModel. """
function is_feasible(gm::GlobalModel)
    tighttol = get_param(gm, :tighttol)
    for bbl in gm.bbls
        is_feasible(bbl, tighttol) || return false
        for ll in bbl.lls
            is_feasible(ll, tighttol) || return false
        end
    end
    return true
end

""" Checks whether a BBL or GM is sampled. """
is_sampled(bbl::BlackBoxLearner) = size(bbl.X,1) != 0

is_sampled(gm::GlobalModel) = all(is_sampled(bbl) for bbl in gm.bbls)

""" Clears all sampling, training and optimization data from GlobalModel. """
function clear_data!(gm::GlobalModel)
    clear_tree_constraints!(gm, gm.bbls)
    clear_data!.(gm.bbls)
    gm.solution_history = DataFrame(string.(gm.vars) .=> [Float64[] for i=1:length(gm.vars)])
end

update_vexity(gm::GlobalModel) = update_vexity.(gm.bbls)

""" Prints relevant details of a GlobalModel. """
function print_details(gm::GlobalModel)
    @info "GlobalModel $(gm.name) has:"
    n_vars = length(gm.vars)
    @info "$(n_vars) variables,"
    all_types = list_of_constraint_types(gm.model)
    l_vartypes = [JuMP.VariableRef, JuMP.GenericAffExpr{Float64, VariableRef}]
    l_constrs = []
    nl_constrs = []
    l_constypes = [MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}]
    for (vartype, constype) in all_types
        constrs_of_type = JuMP.all_constraints(gm.model, vartype, constype)
        if any(vartype .== l_vartypes) && any(constype .== l_constypes)
            append!(l_constrs, constrs_of_type)
        else
            append!(nl_constrs, constrs_of_type)
        end
    end
    @info "$(length(l_constrs)) linear constraints."
    @info "$(length(nl_constrs)) nonlinear constraints."

    n_eqs = length(findall(x -> x.equality, gm.bbls))
    n_ineqs = length(gm.bbls) - n_eqs
    obj_bbl = filter(x -> x.dependent_var == gm.objective, 
        [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])
    @info "$(n_ineqs - length(obj_bbl)) black box inequalities,"
    @info "$(n_eqs) black box equalities,"
    n_lls = sum(length(bbl.lls) for bbl in gm.bbls)
    if n_lls > 0
        @info "$(n_lls) linked constraints,"
    end
    if isempty(obj_bbl)
        @info "And a linear objective."
    else
        @info "And a nonlinear objective."
    end
    return
end    
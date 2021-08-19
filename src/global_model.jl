"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to use add_nonlinear_constraint to add bbls.
model must be a mixed integer convex model.
nonlinear_model can contain JuMP.NonlinearConstraints.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                            # Associated JuMP.Model
    name::String = "Model"                                       # Name
    bbls::Array{BlackBoxLearner} = BlackBoxLearner[]             # Constraints to be learned
    vars::Array{JuMP.VariableRef} = JuMP.all_variables(model)    # JuMP variables
    objective = JuMP.objective_function(model)                # Original objective function
    solution_history::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars)) # Solution history
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
    JuMP.all_variables(bbo::Union{GlobalModel, BlackBoxLearner})
    JuMP.all_variables(bbls::Array{BlackBoxLearner})

Extends JuMP.all_variables to GlobalModels and BlackBoxLearners. 
TODO: add ability to add variables to GlobalModels. 
"""
JuMP.all_variables(bbo::Union{GlobalModel, BlackBoxLearner}) = bbo.vars

function JuMP.all_variables(bbls::Array{BlackBoxLearner})
    return unique(Iterators.flatten(([JuMP.all_variables(bbl) for bbl in bbls])))
end

""" Extends JuMP.set_optimizer to GlobalModels. """
function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    JuMP.set_optimizer(gm.model, optimizer_factory)
end

"""
    get_bounds(model::Union{JuMP.Model, BlackBoxLearner, Array{BlackBoxLearner})

Returns bounds of all variables.
"""
function get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxLearner, 
                                 Array{BlackBoxLearner}})
    return get_bounds(JuMP.all_variables(model))
end

"""
    get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxLearner, 
                 Array{BlackBoxLearner}})

Returns only unbounded variables. 
"""
function get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxLearner, 
                      Array{BlackBoxLearner}})
    return get_unbounds(JuMP.all_variables(model))
end

"""
    determine_vars(gm::GlobalModel,
                        constraint::Union{JuMP.ConstraintRef, Expr};
                        vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                        expr_vars::Union{Nothing, Array} = nothing)

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

 Adds a new nonlinear constraint to Global Model. Standard method for adding bbls.
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
        !isnothing(dependent_var) && throw(OCTException("Constraint " * name * " is of type $(string(typeof(constraint))) " *
                                                        "and cannot have a dependent variable " * string(dependent_var) * "."))
        new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        equality = equality, name = name)
        set_param(new_bbl, :n_samples, Int(ceil(get_param(gm, :sample_coeff)*sqrt(length(vars)))))
        push!(gm.bbls, new_bbl)
        JuMP.delete(gm.model, constraint)   
    end
end

"""
    add_nonlinear_or_compatible(gm::GlobalModel,
                         constraint::Union{JuMP.ConstraintRef, Expr};
                         vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                         expr_vars::Union{Nothing, Array} = nothing,
                         dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                         name::String = gm.name * "_" * string(length(gm.bbls) + 1),
                         equality::Bool = false)

Extents add_nonlinear_constraint to recognize JuMP compatible constraints and add them
as normal JuMP constraints
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
                    @constraint(gm.model, constr_expr == dependent_var)
                elseif equality
                    @constraint(gm.model, constr_expr == 0)
                elseif !isnothing(dependent_var)
                    @constraint(gm.model, constr_expr >= dependent_var)
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
    add_linked_constraint(bbc::BlackBoxClassifier, linked_vars::Array{JuMP.Variable})
    add_linked_constraint(bbr::BlackBoxRegressor, linked_vars::Array{JuMP.Variable}, linked_dependent::JuMP.Variable)

Adds variables that obey the same constraint structure. 
Use in case when a nonlinear constraint is repeated more than once, so that the underlying
approximator is replicated without rebuilding the tree approximation. 
Note that the bounds used for sampling are for the original variables!!
"""
function add_linked_constraint(gm::GlobalModel, bbc::BlackBoxClassifier, vars::Array{JuMP.VariableRef})
    length(vars) == length(bbc.vars) || throw(OCTException("BBC $(bbc.name) does not" *
    " have the same number of variables as linked variables $(vars)."))
    if !isempty(bbc.mi_constraints)
        clear_tree_constraints!(gm, bbc)
        @info "Cleared constraints from BBC $(bbc.name) since it was relinked."
    end
    push!(bbc.lls, LinkedClassifier(vars = vars))
    return
end

function add_linked_constraint(gm::GlobalModel, bbr::BlackBoxRegressor, vars::Array{JuMP.VariableRef}, 
                                dependent_var::JuMP.VariableRef)
    length(vars) == length(bbr.vars) || throw(OCTException("BBR $(bbr.name) does not" *
    " have the same number of variables as linked variables $(vars)"))
    if !isempty(bbr.mi_constraints)
        clear_tree_constraints!(gm, bbr)
        @info "Cleared constraints from BBR $(bbr.name) since it was relinked."
    end
    push!(bbr.lls, LinkedRegressor(vars = vars, dependent_var = dependent_var))
    return
end

"""
    nonlinearize!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    nonlinearize!(gm::GlobalModel)

Turns gm.model into the nonlinear representation.
NOTE: to get back to MI-compatible forms, must rebuild model from scratch.
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

"""Separates and returns linear and nonlinear constraints in a model. """
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
        throw(OCTException(string("BlackBoxClassifier ", bbc.name, " has not been trained yet.")))
    else
        return bbc.accuracies[end]
    end
end

function evaluate_accuracy(bbr::BlackBoxRegressor)
    if bbr.convex && !bbr.equality
        return 1.
    elseif isempty(bbr.learners)
        throw(OCTException(string("BlackBoxRegressor ", bbr.name, " has not been trained yet.")))
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
    feas_gap(gm::GlobalModel)

Evaluates relative feasibility gap at solution. 
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

""" Clears all sampling, training and optimization data from GlobalModel."""
function clear_data!(gm::GlobalModel)
    clear_tree_constraints!(gm, gm.bbls)
    clear_data!.(gm.bbls)
    gm.solution_history = DataFrame([Float64 for i=1:length(gm.vars)], string.(gm.vars))
end

update_vexity(gm::GlobalModel) = update_vexity.(gm.bbls)
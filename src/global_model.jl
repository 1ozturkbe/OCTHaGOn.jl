"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to use add_nonlinear_constraint to add bbfs.
model must be a mixed integer convex model.
nonlinear_model can contain JuMP.NonlinearConstraints.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                            # JuMP model
    name::String = "Model"                                       # Example name
    bbfs::Array{BlackBoxClassifier, BlackBoxRegressor} = []      # Black box (>/= 0) functions
    vars::Array{VariableRef} = JuMP.all_variables(model)         # JuMP variables
    solution_history::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars)) # Solution history
    params::Dict = gm_defaults()                 # GM settings
end

function Base.show(io::IO, gm::GlobalModel)
    println(io, "GlobalModel " * gm.name * " with $(length(gm.vars)) variables: ")
    println(io, "Has $(length(gm.bbfs)) BlackBoxLearners.")
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

set_param(gm::GlobalModel, key::Symbol, val) = set_param(gm.params, key, val)
get_param(gm::GlobalModel, key::Symbol) = get_param(gm.params, key)

"""
    (gm::GlobalModel)(name::String)

Finds BlackBoxLearner in GlobalModel by name.
"""
function (gm::GlobalModel)(name::String)
    fn_names = getfield.(gm.bbfs, :name)
    fns = gm.bbfs[findall(x -> x == name, fn_names)]
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

"""
    JuMP.all_variables(bbo::Union{GlobalModel, BlackBoxClassifier, BlackBoxRegressor})
    JuMP.all_variables(bbfs::Array{BlackBoxClassifier, BlackBoxRegressor})

Extends JuMP.all_variables to GlobalModels and BlackBoxLearners. 
TODO: add ability to add variables to GlobalModels. 
"""
JuMP.all_variables(bbo::Union{GlobalModel, BlackBoxClassifier, BlackBoxRegressor}) = bbo.vars

function JuMP.all_variables(bbfs::Array{BlackBoxClassifier, BlackBoxRegressor})
    return unique(Iterators.flatten(([JuMP.all_variables(bbf) for bbf in bbfs])))
end

""" Extends JuMP.set_optimizer to GlobalModels. """
function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    JuMP.set_optimizer(gm.model, optimizer_factory)
end

"""
    get_bounds(model::Union{JuMP.Model, BlackBoxClassifier, BlackBoxRegressor, 
                            Array{BlackBoxClassifier, BlackBoxRegressor})

Returns bounds of all variables.
"""
function get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxClassifier, BlackBoxRegressor, 
                                 Array{BlackBoxClassifier, BlackBoxRegressor}})
    return get_bounds(JuMP.all_variables(model))
end

"""
    get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxClassifier, BlackBoxRegressor, 
                 Array{BlackBoxClassifier, BlackBoxRegressor}})

Returns only unbounded variables. 
"""
function get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxClassifier, BlackBoxRegressor, 
                      Array{BlackBoxClassifier, BlackBoxRegressor}})
    return get_unbounds(JuMP.all_variables(model))
end

"""
    determine_vars(gm::GlobalModel,
                        constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                        vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                        expr_vars::Union{Nothing, Array} = nothing)

Takes on parsing and allocation of variables depending on user input. 
"""
function determine_vars(gm::GlobalModel,
                        constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                        vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                        expr_vars::Union{Nothing, Array} = nothing)
    if constraint isa Union{JuMP.ScalarConstraint, JuMP.ConstraintRef}
        # Since we cannot determine variables inside JuMP constraints yet...
        return vars, expr_vars
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
                     constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = gm.name * " " * string(length(gm.bbfs) + 1),
                     equality::Bool = false)

 Adds a new nonlinear constraint to Global Model. Standard method for adding BBFs.
"""
function add_nonlinear_constraint(gm::GlobalModel,
                     constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = gm.name * "_" * string(length(gm.bbfs) + 1),
                     equality::Bool = false)
    vars, expr_vars = determine_vars(gm, constraint, vars = vars, expr_vars = expr_vars)
    if constraint isa Expr
        if isnothing(dependent_var)
            new_bbf = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                         equality = equality, name = name)
            push!(gm.bbfs, new_bbf)
            return
        else
            new_bbf = BlackBoxRegressor(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        dependent_var = dependent_var, equality = equality, name = name)
            push!(gm.bbfs, new_bbf)
            return
        end
    elseif constraint isa Union{JuMP.ScalarConstraint, JuMP.ConstraintRef}
        !isnothing(dependent_var) && throw(OCTException("Constraint " * name * " is of type $(string(typeof(constraint))) " *
                                                        "and cannot have a dependent variable " * string(dependent_var) * "."))
        if constraint isa JuMP.ScalarConstraint
            con = JuMP.add_constraint(gm.model, constraint)
            JuMP.delete(gm.model, con)
            new_bbf = BlackBoxClassifier(constraint = con, vars = vars, expr_vars = expr_vars,
                                         equality = equality, name = name)
            push!(gm.bbfs, new_bbf)
            return
        else
            JuMP.delete(gm.model, constraint)   
            new_bbf = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                         equality = equality, name = name)
            push!(gm.bbfs, new_bbf)
            return     
        end
end

"""
    add_nonlinear_or_compatible(gm::GlobalModel,
                         constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                         vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                         expr_vars::Union{Nothing, Array} = nothing,
                         dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                         name::String = gm.name * "_" * string(length(gm.bbfs) + 1),
                         equality::Bool = false)

Extents add_nonlinear_constraint to recognize JuMP compatible constraints and add them
as normal JuMP constraints
"""
function add_nonlinear_or_compatible(gm::GlobalModel,
                     constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = gm.name * "_" * string(length(gm.bbfs) + 1),
                     equality::Bool = false)
    vars, expr_vars = determine_vars(gm, constraint, vars = vars, expr_vars = expr_vars)
    fn = functionify(constraint)
    if fn isa Function
        try
            constr_expr = Base.invokelatest(fn, expr_vars...)
            if equality && !isnothing(dependent_var)
                @constraint(gm.model, constr_expr == dependent_var)
            elseif equality
                @constraint(gm.model, constr_expr == 0)
            elseif !isnothing(dependent_var)
                @constraint(gm.model, constr_expr >= dependent_var)
            else 
                @constraint(gm.model, constr_expr >= 0)
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
    nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxClassifier, BlackBoxRegressor})
    nonlinearize!(gm::GlobalModel)

Turns gm.model into the nonlinear representation.
NOTE: to get back to MI-compatible forms, must rebuild model from scratch.
"""
function nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxClassifier,BlackBoxRegressor})
    for (i, bbf) in enumerate(bbfs)
        if bbf.constraint isa JuMP.ConstraintRef
            JuMP.add_constraint(gm.model, bbf.constraint)
        elseif bbf.constraint isa Expr
            symb = Symbol(bbf.name)
            vars = flat(bbf.expr_vars) # We want flattening of dense vars.
            var_ranges = []
            count = 0
            for varlist in bbf.expr_vars
                if varlist isa VariableRef
                    count += 1
                    push!(var_ranges, count)
                else
                    push!(var_ranges, (count + 1 : count + length(varlist)))
                    count += length(varlist)
                end
            end
            expr = bbf.constraint
            flat_expr = :((x...) -> $(expr)([x[i] for i in $(var_ranges)]...))
            fn = eval(flat_expr)
            JuMP.register(gm.model, symb, length(vars), fn; autodiff = true)
            expr = Expr(:call, symb, vars...)
            if bbf.equality && bbf isa BlackBoxRegressor
                JuMP.add_NL_constraint(gm.model, :($(expr) == $(bbf.dependent_var)))
            elseif bbf.equality
                JuMP.add_NL_constraint(gm.model, :($(expr) == 0))
            elseif bbf isa BlackBoxRegressor
                JuMP.add_NL_constraint(gm.model, :($(expr) <= $(bbf.dependent_var)))
            else
                JuMP.add_NL_constraint(gm.model, :($(expr) >= 0))
            end
        end
    end
    return
end

function nonlinearize!(gm::GlobalModel)
    nonlinearize!(gm, gm.bbfs)
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

""" Returns the feasibility of data points in a BBF or GM. """
feasibility(bbc::BlackBoxClassifier) = bbf.feas_ratio
feasibility(bbr::BlackBoxClassifier) = size(bbr.X, 1) / (size(bbr.X,1) + size(bbr.infeas_X, 1))
feasibility(bbls::Union{BlackBoxClassifier, BlackBoxRegressor}) = feasibility.(bbls)
feasibility(gm::GlobalModel) = feasibility.(gm.bbfs)

""" Returns the accuracy of learners in a BBF or GM. """
function accuracy(bbc::BlackBoxClassifier)
    if bbc.feas_ratio in [1., 0]
        @warn(string("Accuracy of BlackBoxClassifier ", bbc.name, " is tautological."))
        return 1.
    elseif isempty(bbc.learners)
        throw(OCTException(string("BlackBoxClassifier ", bbc.name, " has not been trained yet.")))
    else
        return bbc.accuracies[end]
    end
end

function accuracy(bbr::BlackBoxRegressor)
    if isempty(bbr.learners)
        throw(OCTException(string("BlackBoxRegressor ", bbr.name, " has not been trained yet.")))
    else
        return bbr.accuracies[end]
    end
end

accuracy(bbfs::Array{BlackBoxClassifier, BlackBoxRegressor}) = accuracy.(bbfs)
accuracy(gm::GlobalModel) = accuracy.(gm.bbfs)

""" 
    JuMP.optimize!(gm::GlobalModel; kwargs...)

Applies JuMP.optimize! to GlobalModels, and saves solution history. 
"""
function JuMP.optimize!(gm::GlobalModel; kwargs...)
    JuMP.optimize!(gm.model, kwargs...)
    append!(gm.solution_history, solution(gm), cols=:setequal)
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

""" Evaluates each constraint at solution to make sure it is feasible. """
function evaluate_feasibility(gm::GlobalModel)
    soln = solution(gm);
    for fn in gm.bbfs
        eval!(fn, soln)
    end
    return [gm.bbfs[i].Y[end] >= 0 for i=1:length(gm.bbfs)]
end

""" Matches BBFs to associated variables (except for dependent variables in Regressors). """
function match_bbfs_to_vars(bbfs::Array,
                            vars::Array = JuMP.all_variables(bbfs))
    # TODO: improve types.
    bbf_to_var = Dict()
    for bbf in bbfs
        unb = false
        for var in vars
            if var in bbf.vars
                if unb
                    push!(bbf_to_var[bbf], var)
                else
                    bbf_to_var[bbf] = [var]
                    unb = true
                end
            end
        end
    end
    # Sorting so that we penalize both in the number of unbounded and total variables.
    bbf_to_var = sort(collect(bbf_to_var), by= x->length(x[2])*length(x[1].vars)^0.5)
    return bbf_to_var
end

match_bbfs_to_vars(gm::GlobalModel, vars::Array = JuMP.all_variables(gm)) = match_bbfs_to_vars(gm.bbfs, vars)

""" Clears all sampling, training and optimization data from GlobalModel."""
function clear_data!(gm::GlobalModel)
    clear_data!.(gm.bbfs)
    gm.solution_history = DataFrame([Float64 for i=1:length(gm.vars)], string.(gm.vars))
end
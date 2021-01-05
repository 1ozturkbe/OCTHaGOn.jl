"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to use add_nonlinear_constraint to add bbfs.
model must be a mixed integer convex model.
nonlinear_model can contain JuMP.NonlinearConstraints.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                            # JuMP model
    name::String = "Model"                                       # Example name
    bbfs::Array{BlackBoxFunction} = Array{BlackBoxFunction}[]    # Black box (>/= 0) functions
    vars::Array{VariableRef} = JuMP.all_variables(model)         # JuMP variables
    solution_history::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars)) # Solution history
    settings::Dict = gm_defaults()                 # GM settings
end

function Base.show(io::IO, gm::GlobalModel)
    println(io, "GlobalModel " * gm.name * " with $(length(gm.vars)) variables: ")
    println(io, "Has $(length(gm.bbfs)) BlackBoxFunctions.")
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

set_param(gm::GlobalModel, key::Symbol, val) = set_param(gm.settings, key, val)
get_param(gm::GlobalModel, key::Symbol) = get_param(gm.settings, key)

"""
    (gm::GlobalModel)(name::String)

Finds BlackBoxFunction in GlobalModel by name.
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
    JuMP.all_variables(gm::Union{GlobalModel, BlackBoxFunction})

Extends JuMP.all_variables to GMs and BBFs
and makes sure that the variables are updated.
"""
function JuMP.all_variables(gm::Union{GlobalModel, BlackBoxFunction})
    if gm isa GlobalModel
        gm.vars = JuMP.all_variables(gm.model)
    end
    return gm.vars
end

function JuMP.all_variables(bbfs::Array{BlackBoxFunction})
    return unique(Iterators.flatten(([all_variables(bbf) for bbf in bbfs])))
end


""" Extends JuMP.set_optimizer to GlobalModels. """
function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    JuMP.set_optimizer(gm.model, optimizer_factory)
end

"""
    get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxFunction})
    get_bounds(bbfs::Array{BlackBoxFunction})

Returns bounds of all variables.
"""
function get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxFunction, Array{BlackBoxFunction, 1}})
    return get_bounds(all_variables(model))
end

"""
    get_unbounds(model::Union{GlobalModel, JuMP.Model, BlackBoxFunction})
    get_unbounds(bbfs::Array{BlackBoxFunction})

Returns bounds of all unbounded variables.
"""
function get_unbounds(model::Union{JuMP.Model, GlobalModel, BlackBoxFunction, Array{BlackBoxFunction}})
    return get_unbounds(all_variables(model))
end

""" Checks outer-boundedness of variables. """
function check_bounds(bounds::Dict)
    if any(isinf.(Iterators.flatten(values(bounds))))
        throw(OCTException("Unbounded variables in model!"))
    else
        return
    end
end

function get_max(a, b)
    return maximum([a,b])
end

function get_min(a,b)
    return minimum([a,b])
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
    if constraint isa JuMP.ScalarConstraint 
        !isnothing(dependent_var) && throw(OCTException("Constraint " * name * " is of type JuMP.ScalarConstraint, " *
                                                        "and cannot have a dependent variable " * string(dependent_var) * "."))
        con = JuMP.add_constraint(gm.model, constraint)
        JuMP.delete(gm.model, con)
        new_bbf = BlackBoxFunction(constraint = con, vars = vars, expr_vars = expr_vars,
                                   dependent_var = dependent_var,
                                   equality = equality, name = name)
        set_param(new_bbf, :n_samples, Int(ceil(get_param(gm, :sample_coefficient)*sqrt(length(vars))))) 
        push!(gm.bbfs, new_bbf)
        return
    end
    if constraint isa JuMP.ConstraintRef
        !isnothing(dependent_var) && throw(OCTException("Constraint " * name * " is of type JuMP.ConstraintRef, " *
        "and cannot have a dependent variable " * string(dependent_var) * "."))
        JuMP.delete(gm.model, constraint)
    end
    new_bbf = BlackBoxFunction(constraint = constraint, vars = vars, expr_vars = expr_vars,
                               dependent_var = dependent_var, 
                               equality = equality, name = name)
    if !isnothing(dependent_var)
        set_param(new_bbf, :regression, true)
    end
    set_param(new_bbf, :n_samples, Int(ceil(get_param(gm, :sample_coefficient)*sqrt(length(vars))))) 
    push!(gm.bbfs, new_bbf)
    return
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
    nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxFunction})
    nonlinearize!(gm::GlobalModel)

Turns gm.model into the nonlinear representation.
NOTE: to get back to MI-compatible forms, must rebuild model from scratch.
"""
function nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxFunction})
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
            if bbf.equality
                JuMP.add_NL_constraint(gm.model, :($(expr) == 0))
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
function feasibility(bbf::Union{GlobalModel, Array{BlackBoxFunction, DataConstraint}, DataConstraint, BlackBoxFunction})
    if isa(bbf, Union{BlackBoxFunction, DataConstraint})
        return bbf.feas_ratio
    elseif isa(bbf, Array{BlackBoxFunction, DataConstraint})
        return [feasibility(fn) for fn in bbf]
    else
        return [feasibility(fn) for fn in bbf.bbfs]
    end
end

""" Returns the accuracy of learners in a BBF or GM. """
function accuracy(bbf::Union{GlobalModel, BlackBoxFunction})
    if isa(bbf, BlackBoxFunction)
        if bbf.feas_ratio in [1., 0]
            @warn(string("Accuracy of BlackBoxFunction ", bbf.name, " is tautological."))
            return 1.
        elseif isempty(bbf.learners)
            throw(OCTException(string("BlackBoxFunction ", bbf.name, " has not been trained yet.")))
        else
            return bbf.accuracies[end]
        end
    else
        return [accuracy(fn) for fn in bbf.bbfs]
    end
end

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
    variables = all_variables(m)
    vals = getvalue.(variables)
    return DataFrame(vals', string.(variables))
end

"""
    save_solution(gm::GlobalModel; dir::String = SAVE_DIR)

Saves the optimal solution of GlobalModel as a CSV.
"""
function save_solution(gm::GlobalModel; name::String = gm.name, dir::String = SAVE_DIR)
    CSV.write(dir * name * ".csv", solution(gm))
end

""" Evaluates each constraint at solution to make sure it is feasible. """
function evaluate_feasibility(gm::GlobalModel)
    soln = solution(gm);
    for fn in gm.bbfs
        eval!(fn, soln)
    end
    return [gm.bbfs[i].Y[end] >= 0 for i=1:length(gm.bbfs)]
end

""" Matches BBFs to associated variables. """
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
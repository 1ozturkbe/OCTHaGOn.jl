"""
    add_variables_from_data!(gm::Union{JuMP.Model, GlobalModel},
                            X::DataFrame)

Adds/finds variables depending on the columns of X. 
"""
function add_variables_from_data!(m::JuMP.Model,
                                 X::DataFrame)
    varkeys = names(X)
    vars = []
    data_vars = []
    for key in varkeys
        try 
            push!(vars, m[Symbol(key)])
        catch KeyError
            nv = @variable(m, base_name = string(key))
            m[Symbol(key)] = nv
            push!(vars, nv)
            push!(data_vars, Symbol(key))
        end
    end
    length(data_vars) != 0 && @info("Added $(length(data_vars)) variables from data: " * string(data_vars))
    return vars
end

add_variables_from_data!(gm::GlobalModel, X::DataFrame) = add_variables_from_data!(gm.model, X)


""" 
    bound_to_data!(gm::Union{JuMP.Model, GlobalModel},
                  X::DataFrame)

Constrains the domain of relevant variables to the box interval defined by X.
"""
function bound_to_data!(m::JuMP.Model,
                       X::DataFrame)
    vars = []
    try 
        vars = [m[Symbol(key)] for key in names(X)]
    catch UndefVarError
        throw(OCTHaGOnException("Please make sure the keys of X match with variables in model $(gm)."))
    end
    ranges = Dict(m[Symbol(key)] => [minimum((X[!, Symbol(key)])), maximum((X[!, Symbol(key)]))] for key in names(X))
    bound!(m, ranges);
    return
end
    
bound_to_data!(gm::GlobalModel, X::DataFrame) = bound_to_data!(gm.model, X)

"""
    add_datadriven_constraint(gm::GlobalModel,
                     X::DataFrame, Y::Array;
                     constraint::Union{Nothing, JuMP.ConstraintRef, Expr} = nothing, 
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = "bbl" * string(length(gm.bbls) + 1),
                     equality::Bool = false)

 Adds a data-driven constraint to GlobalModel. Data driven BBLs
do not allow of resampling. 
"""
function add_datadriven_constraint(gm::GlobalModel,
                     X::DataFrame, Y::Array;
                     constraint::Union{Nothing, JuMP.ConstraintRef, Expr} = nothing, 
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = "bbl" * string(length(gm.bbls) + 1),
                     equality::Bool = false)
    if isnothing(vars)
        vars = add_variables_from_data!(gm, X)
    else
        @assert size(X, 2) == length(vars)
    end
    expr_vars = vars
    if isnothing(dependent_var)
        new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        equality = equality, name = name)
        add_data!(new_bbl, X, Y)
        push!(gm.bbls, new_bbl)
        return
    else
        new_bbl = BlackBoxRegressor(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                    dependent_var = dependent_var, equality = equality, name = name)
        add_data!(new_bbl, X, Y)
        push!(gm.bbls, new_bbl)
        return
    end
end
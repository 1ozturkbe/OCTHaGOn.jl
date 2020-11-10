#=
on_jumpmodels:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-25
=#

# NOTE: We have to circumvent JuMP.NLexpressions to be able to support nonlinear expressions
#       without breaking everything...
#       Thus why all operations on JuMP-incompatible functions is on Exprs.


"""
    fetch_variable(model::JuMP.Model, varkey::Union{Symbol, String, VariableRef, Array})

Returns JuMP.VariableRefs that match a given Symbol, String, VariableRef,
or array of these.
"""
function fetch_variable(model::JuMP.Model, varkey::Union{Symbol, String, VariableRef, Array})
    if varkey isa Array
        return [fetch_variable(model, key) for key in varkey]
    end
    if varkey isa Symbol
        return model[varkey]
    elseif varkey isa VariableRef
        is_valid(model, varkey) && return varkey
        throw(KeyError(string("Varkey ", varkey,
                              " is invalid for Model.")))
    elseif varkey isa String
        return JuMP.variable_by_name(model, varkey)
    end
end

"""
    get_bounds(vars::Array{VariableRef})

Returns bounds of JuMP variables.
"""
function get_bounds(vars::Array{VariableRef})
    bounds = Dict(var => [-Inf, Inf] for var in vars);
    for var in vars
        if JuMP.has_lower_bound(var)
            bounds[var][1] = JuMP.lower_bound(var);
        end
        if JuMP.has_upper_bound(var)
            bounds[var][2] = JuMP.upper_bound(var);
        end
    end
    return bounds
end

"""
    data_to_DataFrame(data::Union{Dict, DataFrame, DataFrameRow})

Gets data with different keys, and returns a DataFrame with string headers.
"""
function data_to_DataFrame(data::Union{Dict, DataFrame, DataFrameRow})
    if data isa DataFrame
        return data
    elseif data isa DataFrameRow
        return DataFrame(data)
    else
        newdata = Dict(string(key) => value for (key, value) in data)
        return DataFrame(newdata)
    end
end


"""
    distance_to_set(val::Union{Array{<:Real},<:Real}, set::MOI.AbstractSet)

Wrapper around MathOptSetDistances.distance_to_set.
Distance 0 if value ∈ set. Otherwise, returns Float64.
"""
function distance_to_set(val::Union{Array{<:Real},<:Real}, set::MOI.AbstractSet)

    return MathOptSetDistances.distance_to_set(MathOptSetDistances.DefaultDistance(), val, set)
end

function get_constant(set::MOI.AbstractSet)
    """Returns constant of MOI.Abstract Set, if it exists..."""
    try
        MOI.constant(set)
    catch
        nothing
    end
end

"""
    functionify(constraint::JuMP.ConstraintRef, vars::Array)
    functionify(constraint::Expr, vars::Array)

Blunt function that returns an "evaluate-able" function from an Expr, or
nothing for a JuMP.ConstraintRef.
Can extend in the future to other elements.
"""
function functionify(constraint)
    if constraint isa Expr
        f = eval(constraint)
        f isa Function && return f
        throw(OCTException(string("functionify", f, "is not a valid function")))
    else
        return nothing
    end
end

"""
    vars_from_expr(expr::Expression, model::JuMP.Model)

Returns the JuMP Variables that are associated with a given function.
Note: Function Expr's must be defined with a single input or a tuple of inputs, eg:

    ex = :(x -> 5*x)
    ex = :((x, y, z) -> sum(x[i] for i=1:4) - y[1] * y[2] + z)
"""
function vars_from_expr(expr::Expr, vars::Array{VariableRef, 1})
    eval(expr) isa Function || throw(OCTException((string("eval of the following
                                     Expr must return a function: ", expr))))
    expr.args[1] isa Symbol && return [vars[1].model[expr.args[1]]]
    return [vars[1].model[outer] for outer in expr.args[1].args]

end

function vars_from_expr(expr::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef}, vars::Array{VariableRef, 1})
    return vars
end

"""
    linearize_objective!(model::JuMP.Model)
Makes sure that the objective function is affine.
"""
function linearize_objective!(model::JuMP.Model)
    objtype = JuMP.objective_function(model)
    objsense = string(JuMP.objective_sense(model))
    if objtype isa Union{VariableRef, GenericAffExpr} || objsense == "FEASIBILITY_SENSE"
        return
    else
        aux = @variable(model)
        @objective(model, Min, aux)
        # Default optimization problem is always a minimization
        coeff = 1;
        objsense == "MAX_SENSE" && (coeff = -1)
        try
            @constraint(model, aux >= coeff*JuMP.objective_function(model))
        catch
            @NLconstraint(model, aux >= coeff*JuMP.objective_function(model))
        end
        return
    end
end
#=
on_jumpmodels:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-25
=#

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

function get_bounds(vars::Array{VariableRef})
    """ Returns bounds of selected variables."""
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

function sanitize_data(data::Union{Dict, DataFrame, DataFrameRow})
    """ Gets data with different keys, and returns a DataFrame with string headers. """
    if data isa DataFrame
        return data
    elseif data isa DataFrameRow
        return DataFrame(data)
    else
        newdata = Dict(string(key) => value for (key, value) in data)
        return DataFrame(newdata)
    end
end

function distance_to_set(val::Union{Array{<:Real},<:Real}, set::MOI.AbstractSet)
    """Wrapper around MathOptSetDistances.distance_to_set.
       Distance 0 if value ∈ set. Otherwise, returns Float64. """
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

function evaluate(constraint::JuMP.ConstraintRef, data::Union{Dict, DataFrame})
    """ Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform. """
    constr_obj = constraint_object(constraint)
    rhs_const = get_constant(constr_obj.set)
    clean_data = sanitize_data(data);
    if size(clean_data, 1) == 1
        val = JuMP.value(constr_obj.func, i -> get(clean_data, string(i), Inf)[1])
        if isinf(val)
            throw(OCTException(string("Constraint ", constraint, " returned an infinite value.")))
        end
        if isnothing(rhs_const)
            return -1*distance_to_set(val, constr_obj.set)
        else
            coeff=1
            if constr_obj.set isa MOI.LessThan
                coeff=-1
            end
            return coeff*(val - rhs_const)
        end
    else
        vals = [evaluate(constraint, DataFrame(clean_data[i,:])) for i=1:size(clean_data,1)]
        return vals
    end
end

function evaluate(constraint::JuMP.NonlinearExpression, data::Union{Dict, DataFrame})
    """ Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform. """
    clean_data = sanitize_data(data);
    if size(clean_data, 1) == 1
        val = JuMP.value(constraint, i -> get(clean_data, string(i), Inf)[1])
        if isinf(val)
            throw(OCTException(string("Constraint ", constraint, " returned an infinite value.")))
        end
        return val
    else
        vals = [evaluate(constraint, DataFrame(clean_data[i,:])) for i=1:size(clean_data,1)]
        return vals
    end
end

function linearize_objective!(model::JuMP.Model)
    """Makes sure that the objective function is affine. """
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

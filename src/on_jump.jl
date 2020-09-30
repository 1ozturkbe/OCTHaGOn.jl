#=
on_jumpmodels:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-25
=#

function fetch_variable(model::JuMP.Model, varkey::Union{Symbol, String, VariableRef})
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

function get_bounds(model::JuMP.Model)
    all_vars = JuMP.all_variables(model)
    bounds = Dict(var => [-Inf, Inf] for var in all_vars);
    for var in all_vars
        if JuMP.has_lower_bound(var)
            bounds[var][1] = JuMP.lower_bound(var);
        end
        if JuMP.has_upper_bound(var)
            bounds[var][2] = JuMP.upper_bound(var);
        end
    end
    return bounds
end

function bound!(model::JuMP.Model,
                bounds::Dict)
    """Adds outer bounds to JuMP Model from dictionary of data. """
    for (key, value) in bounds
        @assert value isa Array && length(value) == 2
        var = fetch_variable(model, key);
        if var isa Array # make sure all elements are bounded.
            for v in var
                bound!(model, Dict(v => value))
            end
        else
            if JuMP.has_lower_bound(var) && JuMP.lower_bound(var) <= minimum(value)
                set_lower_bound(var, minimum(value))
            else
                set_lower_bound(var, minimum(value))
            end
            if JuMP.has_upper_bound(var) && JuMP.upper_bound(var) >= maximum(value)
                set_upper_bound(var, maximum(value))
            else
                set_upper_bound(var, maximum(value))
            end
        end
    end
    return
end

# function get_variables(model::JuMP.Model, constraint)


# model = Model()
# @variable(model, x)
# c = @constraint(model, 2x + 1 <= 0)
# obj = constraint_object(c)
# obj.func  # The function

function sanitize_data(model::JuMP.Model, data::Union{Dict, DataFrame})
    """ Gets data with different keys, and returns a DataFrame with string headers. """
    if data isa DataFrame
        return data
    else
        newdata = Dict(string(key) => value for (key, value) in data)
        return DataFrame(newdata)
    end
end

function evaluate(model::JuMP.Model, constraint, data::Union{Dict, DataFrame})
    """ Evaluates a constraint on data in the variables.
        Note that the keys of the Dict have to be uniform. """
    if data isa Dict
        clean_data = sanitize_data(model, data);
        evaluate(model, constraint, clean_data);
    else
        if size(data, 1) == 1
            return JuMP.value(constraint, i -> get(clean_data, i, Inf)[1])
        else
            return value(constraint, i -> get(clean_data, i, Inf))
end
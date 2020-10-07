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

function get_bounds(model::JuMP.Model)
    """ Returns bounds of all variables from a JuMP.Model. """
    all_vars = JuMP.all_variables(model)
    return get_bounds(all_vars)
end

function check_bounds(bounds::Dict)
    """ Checks outer-boundedness. """
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

function check_infeasible_bounds(model::JuMP.Model, bounds::Dict)
    all_bounds = get_bounds(model);
    lbs_model = Dict(key => minimum(value) for (key, value) in all_bounds)
    ubs_model = Dict(key => maximum(value) for (key, value) in all_bounds)
    lbs_bounds = Dict(key => minimum(value) for (key, value) in bounds)
    ubs_bounds = Dict(key => maximum(value) for (key, value) in bounds)
    nlbs = merge(get_max, lbs_model, lbs_bounds)
    nubs = merge(get_min, ubs_model, ubs_bounds)
    if any([nlbs[var] .> nubs[var] for var in keys(nlbs)])
        throw(OCTException("Infeasible bounds."))
    end
    return
end

function bound!(model::JuMP.Model,
                bounds::Dict)
    """Adds outer bounds to JuMP Model from dictionary of data. """
    check_infeasible_bounds(model, bounds)
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
            elseif !JuMP.has_lower_bound(var)
                set_lower_bound(var, minimum(value))
            end
            if JuMP.has_upper_bound(var) && JuMP.upper_bound(var) >= maximum(value)
                set_upper_bound(var, maximum(value))
            else !JuMP.has_upper_bound(var)
                set_upper_bound(var, maximum(value))
            end
        end
    end
    return
end


# model = Model()
# @variable(model, x)
# c = @constraint(model, 2x + 1 <= 0)
# obj = constraint_object(c)
# obj.func  # The function

function sanitize_data(model::JuMP.Model, data::Union{Dict, DataFrame, DataFrameRow})
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

function evaluate(constraint::ConstraintRef, data::Union{Dict, DataFrame})
    """ Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform. """
    constr_obj = constraint_object(constraint)
    rhs_const = get_constant(constr_obj.set)
    clean_data = sanitize_data(constraint.model, data);
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

# function find_variables(sc::ScalarConstraint)
#     """Returns the variables in a ScalarConstraint (Note: NOT NONLINEAR) object. """
#     vars = Array{VariableRef}[];
#     for (var, coef) in sc.func.terms
#         if var isa VariableRef
#             push!(vars, var)
#         end
#         if coeff
#     end
#     return vars
# end


function classify_constraints(model::JuMP.Model)
    """Separates and returns linear and nonlinear constraints in a model. """
    all_types = list_of_constraint_types(model)
    nl_constrs = [];
    l_constrs = [];
    l_vartypes = [JuMP.VariableRef, JuMP.GenericAffExpr{Float64, VariableRef}]
    l_constypes = [MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}]
    for (vartype, constype) in all_types
        constrs_of_type = JuMP.all_constraints(model, vartype, constype)
        if any(vartype .== l_vartypes) && any(constype .== l_constypes)
            append!(l_constrs, constrs_of_type)
        else
            append!(nl_constrs, constrs_of_type)
        end
    end
    if !isnothing(model.nlp_data)
        append!(nl_constrs, model.nlp_data.nlconstr)
    end
    return l_constrs, nl_constrs
end
#=
on_jumpmodels:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-25
=#

# NOTE: We have to circumvent JuMP.NLexpressions to be able to support nonlinear expressions
#       without breaking everything...

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

function substitute(e::Expr, pair)
    MacroTools.prewalk(e) do s
        s == pair.first && return pair.second
        s
    end
end

function substitute_expr(expr::Expr, data::Dict)
    new_expr = copy(expr)
    for (key, value) in data
        new_expr = substitute(new_expr, :($key) => value)
    end
    return eval(expr)
end

# get_locals and _get_outers ripped from here:
#https://discourse.julialang.org/t/determine-the-scope-of-symbols-in-an-expr/47993/2
function get_locals(ex::Expr)
    vars = (solve_from_local ∘ simplify_ex)(ex).args[1].bounds
    map(x -> x.name, vars)
end

_get_outers(_) = Symbol[]
_get_outers(x::Var) = x.is_global ? [x.name] : Symbol[]
function _get_outers(ex::Expr)
    @match ex begin
        Expr(:(=), _, rhs) => _get_outers(rhs)
        Expr(:tuple, _..., Expr(:(=), _, rhs)) => _get_outers(rhs)
        Expr(_, args...) => mapreduce(_get_outers, vcat, args)
    end
end

get_outers(ex) = (unique! ∘ _get_outers ∘ solve_from_local ∘ simplify_ex)(ex)

function outers_to_vars(outers::Array{Symbol}, model::JuMP.Model)
    """ Finds the non-local variables in Expression. """
    vars = Dict()
    for outer in outers
        try
            vars[outer] = model[outer]
        catch
            continue
        end
    end
    return vars
end

function get_bounds(vars::Array{VariableRef})
    """ Returns bounds of JuMP variables."""
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

function data_to_DataFrame(data::Union{Dict, DataFrame, DataFrameRow})
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

function data_to_Dict(data::Union{Dict, DataFrame, DataFrameRow}, model::JuMP.Model)
    """ Turns data into a Dict, with JuMP.VariableRef subs. """
    if data isa Dict
        return Dict(fetch_variable(model, key) => value for (key, value) in data)
    else
        colnames = names(data)
        return Dict(fetch_variable(model, name) => data[!, name] for name in colnames)
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
    clean_data = data_to_DataFrame(data);
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

function evaluate(constraint::Expr, data::Union{Dict, DataFrame})
    """ Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform. """
    clean_data = data_to_DataFrame(data);
    if size(clean_data, 1) == 1
        val = substitute_expr(constraint, data)
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

# NOTE: We have to circumvent JuMP.NLexpressions to be able to support nonlinear expressions
#       without breaking everything...
#       Thus why all operations on JuMP-incompatible functions is on Exprs.


"""
    fetch_variable(model::JuMP.Model, varkey::Union{Symbol, String, VariableRef, Array})

Returns JuMP.VariableRefs that match a given Symbol, String, VariableRef,
or array of these.
"""
function fetch_variable(model::JuMP.Model, varkey::Union{Symbol, String, VariableRef})
    if varkey isa Symbol
        try
            return model[varkey]
        catch UndefVarError
            throw(KeyError(string("Varkey ", varkey,
                              " is invalid for Model.")))
        end
    elseif varkey isa VariableRef
        is_valid(model, varkey) && return varkey
        throw(KeyError(string("Varkey ", varkey,
                              " is invalid for Model.")))
    elseif varkey isa String
        ret = JuMP.variable_by_name(model, varkey)
        !isnothing(ret) && return ret
        throw(KeyError(string("Varkey ", varkey,
                              " is invalid for Model.")))
    end
end

function fetch_variable(model::JuMP.Model, varkey::Array)
    return [fetch_variable(model, key) for key in varkey]
end

""" Helper function for finding bounds. """
function get_bound(var::JuMP.VariableRef)
    if JuMP.has_lower_bound(var)
        if JuMP.has_upper_bound(var)
            return [JuMP.lower_bound(var), JuMP.upper_bound(var)]
        else
            return [JuMP.lower_bound(var), Inf]
        end
    else
        if JuMP.has_upper_bound(var)
            return [-Inf, JuMP.upper_bound(var)]
        else
            return [-Inf, Inf]
        end
    end
end

"""
    get_bounds(var::JuMP.VariableRef})
    get_bounds(vars::Array{JuMP.VariableRef})

Returns bounds of JuMP variables.
"""
function get_bounds(var::JuMP.VariableRef)
    return var => get_bound(var)
end

get_bounds(vars::Array{JuMP.VariableRef}) = Dict(get_bounds(var) for var in vars)

"""
    get_unbounds(var::JuMP.VariableRef)
    get_unbounds(vars::Array{VariableRef})
    get_unbounds(gm::Union{JuMP.Model, GlobalModel})

Returns variables with no lower and/or upper bounds.
"""
function get_unbounds(var::JuMP.VariableRef)
   if JuMP.has_lower_bound(var) 
        if !JuMP.has_upper_bound(var)
            return var => [JuMP.lower_bound(var), Inf]
        else
            return
        end
    else
        if JuMP.has_upper_bound(var)
            return var => [-Inf, JuMP.upper_bound(var)]
        else
            return var => [-Inf, Inf]
        end
    end
end

function get_unbounds(vars::Array{JuMP.VariableRef})
     unbounds = Dict()
     for var in vars
        res = get_unbounds(var)
        if !isa(res, Nothing)
            unbounds[res.first] = res.second
        end
    end
    if length(unbounds) == 0
        return
    else
        return unbounds
    end
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
        newdata = Dict(string(key) => val for (key, val) in data)
        return DataFrame(newdata)
    end
end


"""
    distance_to_set(val::Union{Array{<:Real},<:Real}, set::MOI.AbstractSet)

Wrapper around MathOptSetDistances.distance_to_set.
Distance 0 if val ∈ set. Otherwise, returns Float64.
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
    functionify(constraint)

Blunt function that returns an "evaluate-able" function from an Expr, or
nothing for a JuMP.ConstraintRef.
Can extend in the future to other elements.
"""
function functionify(constraint)
    if constraint isa Expr
        f = eval(constraint)
        f isa Function && return f
        throw(OCTException(string("functionify(", f, ") is not a valid function.")))
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
function vars_from_expr(expr::Expr, model::JuMP.Model)
    if expr.args[1] isa Symbol
        return [fetch_variable(model, expr.args[1])]
    else
        return [fetch_variable(model, arg) for arg in expr.args[1].args]
    end
end

"""
    vars_from_constraint(con::JuMP.ConstraintRef)

Returns the JuMP Variables that are associated with a JuMP.ConstraintRef. 
Note: Currently only works for affine and quadratic constraints. 
"""
function vars_from_constraint(con::JuMP.ConstraintRef)
    confunc = constraint_object(con).func
    confunc isa Union{JuMP.GenericAffExpr, JuMP.GenericQuadExpr} || 
        throw(OCTException("Only affine or quadratic constraints are current supported."))
    vars = [term for (term,val) in confunc.aff.terms]
    for (term, val) in confunc.terms
        append!(vars, [term.a, term.b])
    end
    return unique(vars)
end

"""
    gradientify(expr::Expr, expr_vars::Array)
    gradientify(expr::JuMP.ConstraintRef, expr_vars::Array)

Turns an expression into a gradient-able (via ForwardDiff), flattened function. 
"""
function gradientify(expr::Expr, expr_vars::Array)
    var_ranges = get_var_ranges(expr_vars)
    gradable_fn = x -> Base.invokelatest(functionify(expr), [x[i] for i in var_ranges]...)
    return x -> ForwardDiff.gradient(gradable_fn, x)
end

function gradientify(con::JuMP.ConstraintRef, expr_vars::Array)
    confunc = constraint_object(con).func
    confunc isa Union{JuMP.GenericQuadExpr, JuMP.GenericAffExpr} || throw(
        OCTException("Currently, only supporting gradients of affine or quadratic JuMP constraints. Please " *
                     "submit your constraint as a Expr instead and try again. "))
    flatvars = flat(expr_vars)
    gradarr = zeros(length(flatvars))
    graddict = Dict()
    for (term, val) in confunc.aff.terms # Affine component
        idx = findall(x -> x == term, flatvars)
        if !isempty(idx)
            gradarr[idx[1]] += val
        end
    end
    for (term, val) in confunc.terms # Quadratic component
        idx1 = findall(x -> x == term.a, flatvars)[1]
        idx2 = findall(x -> x == term.b, flatvars)[1]
        graddict[(idx1, idx2)] = val
    end
    gradfn = let gdict = graddict, garr = gradarr
        function (x)
            quad_terms = zeros(length(garr))
            for (term, val) in gdict
                quad_terms[term[1]] = val * x[term[2]]
                quad_terms[term[2]] = val * x[term[1]]
            end
            return garr.*x + quad_terms
        end
    end
    return gradfn
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

""" Checks whether any defined bounds are infeasible by given Model. """
function check_infeasible_bound(bound::Pair)
    key = bound.first
    val = bound.second
    @assert key isa JuMP.VariableRef
    @assert val isa Array
    model_bounds = get_bound(key)
    if minimum(model_bounds) >= maximum(val) || maximum(model_bounds) <= minimum(val)
        throw(OCTException("Infeasible bounds."))
    else
        return [maximum([minimum(model_bounds), minimum(val)]),
                    minimum([maximum(model_bounds), maximum(val)])]
    end
end

"""
    bound!(model::JuMP.Model, bound::Pair)
    bound!(model::JuMP.Model, bounds::Dict)
    bound!(model::GlobalModel, bounds::Union{Pair,Dict})

Adds outer bounds to JuMP Model from Dict or Pair of data.
"""
function bound!(model::JuMP.Model, bound::Pair)
    if isa(bound.first, JuMP.VariableRef)
        tightest_bound = check_infeasible_bound(bound)
        b_min = minimum(tightest_bound)
        b_max = maximum(tightest_bound)
        !isinf(b_min) && JuMP.set_lower_bound(bound.first, b_min)
        !isinf(b_max) && JuMP.set_upper_bound(bound.first, b_max)
    else
        vars = fetch_variable(model, bound.first)
        if vars isa JuMP.VariableRef
            bound!(model, vars => bound.second)
        elseif vars isa Array
            for var in vars
                bound!(model, var => bound.second)
            end
        else
            throw(OCTException("Bound with fetch_variable has failed. Try bounding using JuMP.VariableRefs!"))
        end
    end
    return
end

function bound!(model::JuMP.Model, bounds::Dict)
    for bd in collect(bounds)
        bound!(model, bd)
    end
end
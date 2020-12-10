#=
expr:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-04
Note: This is a dump file when BBFs used to take raw expressions instead of Expr(Function)'s
=#

"""
Generates a map for the auxiliary variables for Expr -> Function conversion.
"""
function generate_auxmap(ex::Expr, vars::Array{JuMP.VariableRef}, model::JuMP.Model)
    outers = get_outers(ex)
    outer_var_mapping = outers_to_vars(outers, model)
    dictmap = Dict()
    count = 1;
    for (key, value) in outer_var_mapping
        if value isa Array
            dictmap[key] = [:(aux[$i]) for i = count:count+length(value)]
            count += length(value)
        else
            count += 1
            dictmap[key] = :(aux[$count])
        end
    end
    new_expr = copy(ex);
    new_expr = substitute_expr(new_expr, dictmap)
    return
end

import MLStyle
import JuliaVariables

"""
    get_locals(ex::Expr)

Get variables in Expr that are locally defined.

get_locals borrowed from here:
https://discourse.julialang.org/t/determine-the-scope-of-symbols-in-an-expr/47993/2
"""
function get_locals(ex::Expr)
    vars = (JuliaVariables.solve_from_local ∘ JuliaVariables.simplify_ex)(ex).args[1].bounds
    map(x -> x.name, vars)
end

_get_outers(_) = Symbol[]
_get_outers(x::JuliaVariables.Var) = x.is_global ? [x.name] : Symbol[]
function _get_outers(ex::Expr)
    MLStyle.@match ex begin
        Expr(:(=), _, rhs) => _get_outers(rhs)
        Expr(:tuple, _..., Expr(:(=), _, rhs)) => _get_outers(rhs)
        Expr(_, args...) => mapreduce(_get_outers, vcat, args)
    end
end

"""
    get_outers(ex::Expr)

Gets the Symbols in the expression that are not local.

get_outers borrowed from here:
https://discourse.julialang.org/t/determine-the-scope-of-symbols-in-an-expr/47993/2
"""
get_outers(ex) = (unique! ∘ _get_outers ∘ JuliaVariables.solve_from_local ∘ JuliaVariables.simplify_ex)(ex)

# Dummy code generating functions from Exprs. Could come in useful.

constr_fn = let lhs = lhsexpr, rhs = rhsexpr, op = op, consts = consts
    if op in [:(=), :>]
        function (x)
            for (key, value) in union(x, consts, sets)
                eval(Meta.parse("$key = $value"))
            end
            lhs_evaled = eval(lhs)
            rhs_evaled = eval(rhs)
            return lhs_evaled-rhs_evaled
        end
    elseif op == :<
        function (x)
            for (key, value) in union(x, consts, sets)
                eval(Meta.parse("$key = $value"))
            end
            lhs_evaled = eval(lhs)
            rhs_evaled = eval(rhs)
            return rhs_evaled-lhs_evaled
        end
    else
        error("unexpected operator ", op)
    end


"""
    substitute_expr(expr::Expr, data::Dict)

Substitutes for the outer variables (symbols from the expr) using data.
"""
function substitute_expr(expr::Expr, data::Dict)
    new_expr = copy(expr)
    for (key, value) in data
        new_expr = substitute(new_expr, :($key) => value)
    end
    return eval(new_expr)
end

"""
    substitute_args is a helper function, by Oscar Dowson.
    See here for more details:
    https://discourse.julialang.org/t/procedural-nonlinear-constaint-generation-string-to-nlconstraint/34799/4
"""
substitute_args(ex, vars) = ex
substitute_args(ex::Symbol, vars) = get(vars, ex, ex)
function substitute_args(ex::Expr, vars)
    for (i, arg) in enumerate(ex.args)
        ex.args[i] = substitute_args(arg, vars)
    end
    return ex
end
#=
expr:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-04
Note: This is a dump file when BBFs used to take raw expressions instead of Expr(Function)'s
=#

function substitute(e::Expr, pair)
    MacroTools.prewalk(e) do s
        s == pair.first && return pair.second
        s
    end
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


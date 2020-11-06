#=
small_scripts:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-04
=#

"""
    chop_dict(data::Dict)

Chops dictionary into sub dictionaries.
Note that it turns scalar var values from Array to Float
for Expr evaluation.
"""
function chop_dict(data::Dict)
    sizes = [size(val, 1) for val in values(data)]
    @assert all(sizes .== maximum(sizes))
    nds = [Dict() for i=1:maximum(sizes)]
    for (key,value) in data
        for i=1:maximum(sizes)
            nds[i][key] = value[i,:]
            if prod(size(nds[i][key])) == 1
                nds[i][key] = value[i,:][1]
            end
        end
    end
    return nds
end

"""
    data_to_Dict(data::Union{Dict, DataFrame, DataFrameRow}, model::JuMP.Model)

Turns data into a Dict, with JuMP.VariableRef subs.
"""
function data_to_Dict(data::Union{Dict, DataFrame, DataFrameRow}, model::JuMP.Model)
    if data isa Dict
        return Dict(fetch_variable(model, key) => value for (key, value) in data)
    else
        colnames = names(data)
        if size(data, 1) == 1
            return Dict(fetch_variable(model, name) => data[!, name][1] for name in colnames)
        else
            return Dict(fetch_variable(model, name) => data[!, name] for name in colnames)
        end
    end
end


# get_locals and _get_outers ripped from here:
#https://discourse.julialang.org/t/determine-the-scope-of-symbols-in-an-expr/47993/2
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

get_outers(ex) = (unique! ∘ _get_outers ∘ JuliaVariables.solve_from_local ∘ JuliaVariables.simplify_ex)(ex)

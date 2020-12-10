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

""" Wrapper around Iterators.flatten. """
flat(arr::Array) = collect(Iterators.flatten(arr))

function substitute(e::Expr, pair)
    MacroTools.prewalk(e) do s
        s == pair.first && return pair.second
        s
    end
end
#=
small_scripts:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-11-04
=#

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

""" Wrapper around Iterators.flatten for variables. """
flat(arr) = Array{VariableRef, 1}(collect(Iterators.flatten(arr)))

""" For substitution into expressions. IMPORTANT. """
function substitute(e::Expr, pair)
    MacroTools.prewalk(e) do s
        s == pair.first && return pair.second
        s
    end
end

""" power function from gams """
power(var, num) = var^num
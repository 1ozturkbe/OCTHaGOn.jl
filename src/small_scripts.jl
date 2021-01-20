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
        return Dict(fetch_variable(model, key) => val for (key, val) in data)
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

""" Checks outer-boundedness of values of a Dict. """
function check_bounds(bounds::Dict)
    if any(isinf.(Iterators.flatten(values(bounds))))
        throw(OCTException("Unbounded variables in model!"))
    else
        return
    end
end

"""
    get_varmap(expr_vars::Array, vars::Array)

Helper function to map vars to flatvars.
Arguments:
    flatvars is a flattened Array{JuMP.VariableRef}
    vars is the unflattened version, usually derived from an Expr.
Returns:
    Dict of ID maps
"""
function get_varmap(expr_vars::Array, vars::Array)
    length(flat(expr_vars)) >= length(vars) || throw(OCTException(string("Insufficiently many input
        variables declared in ", vars, ".")))
    unique(vars) == vars || throw(OCTException(string("Nonunique variables among ", vars, ".")))
    varmap = Tuple[(0,0) for i=1:length(vars)]
    for i = 1:length(expr_vars)
        if expr_vars[i] isa JuMP.VariableRef
            try
                varmap[findall(x -> x == expr_vars[i], vars)[1]] = (i,0)
            catch
                throw(OCTException(string("Scalar variable ", expr_vars[i], " was not properly declared in vars.")))
            end
        else
            for j=1:length(expr_vars[i])
                try
                    varmap[findall(x -> x == expr_vars[i][j], vars)[1]] = (i,j)
                catch
                    continue
                end
            end
        end
    end
    length(varmap) == length(vars) || throw(OCTException(string("Could not properly map
                                            expr_vars: ", expr_vars,
                                            "to vars: ", vars, ".")))
    return varmap
end

get_varmap(expr_vars::Nothing, vars::Array) = nothing

"""Returns the relevant ranges for variables in expr_vars..."""
function get_var_ranges(expr_vars::Array)            
    var_ranges = []
    count = 0
    for varlist in expr_vars
        if varlist isa VariableRef
            count += 1
            push!(var_ranges, count)
        else
            push!(var_ranges, (count + 1 : count + length(varlist)))
            count += length(varlist)
        end
    end
    return var_ranges
end

"""
    infarray(varmap::Array)

Creates a template array for deconstruct function.
"""
function infarray(varmap::Array)
    arr = []
    # Pre allocate and fill!
    copied_map = sort(deepcopy(varmap))
    firsts = first.(copied_map)
    seconds = last.(copied_map)
    arr_length = length(unique(firsts))
    for i=1:arr_length
        tup_idx = findall(j -> j[1] == i, copied_map)
        if seconds[tup_idx] == [0]
            push!(arr, Inf)
        else
            push!(arr, Inf*ones(maximum(seconds[tup_idx])))
        end
    end
    return arr
end

"""
    deconstruct(data::DataFrame, vars::Array, varmap::Array)

Takes in data for input into a Function, and rips it apart into appropriate arrays.
"""
function deconstruct(data::DataFrame, vars::Array, varmap::Array)
    n_samples, n_vars = size(data)
    infarr = infarray(varmap)
    arrs = [];
    stringvars = string.(vars)
    for i = 1:n_samples
        narr = deepcopy(infarr)
        for j = 1:length(varmap)
            if varmap[j] isa Tuple && varmap[j][2] != 0
                narr[varmap[j][1]][varmap[j][2]] = data[i, stringvars[j]]
            else
                narr[varmap[j][1]] = data[i, stringvars[j]]
            end
        end
        push!(arrs, narr)
    end
    return arrs
end
#=
gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-10
=#

include("../../../GAMSFiles.jl/src/GAMSFiles.jl")
using .GAMSFiles
using IterTools

filename = "data/baron_nc_ns/problem3.13.gms"

function constants(gams::Dict{String, Any})
    """Returns all constants from GAMS Model. """
    consts = Dict(); #
    for p in ("parameters", "tables")
        if haskey(gams, p)
            for (k, v) in gams[p]
                v = v isa Ref ? v[] : v
                consts[Symbol(GAMSFiles.getname(k))] = v;
            end
        end
    end
    return consts
end

function constants_to_global(gams)
    consts = constants(gams)
    for (key, value) in consts
        eval(Meta.parse("$(key) = $(value)"))
    end
end

function eq_to_fn(eq, sets, constants)
    @assert(eq isa GAMSFiles.GCall && length(eq.args)==2)
    lhs, rhs, op = eq.args[1], eq.args[2], GAMSFiles.eqops[GAMSFiles.getname(eq)]
    lhsexpr = convert(Expr, lhs)
    rhsexpr = convert(Expr, rhs)
    GAMSFiles.replace_reductions!(lhsexpr, sets)
    GAMSFiles.replace_reductions!(rhsexpr, sets)
    constr_fn = let lhs = lhsexpr, rhs = rhsexpr, op = op, constants = constants
        if op in [:(=), :>]
            function (x)
                for (key, value) in union(x, constants, sets)
                    eval(Meta.parse("$key = $value"))
                end
                lhs_evaled = eval(lhs)
                rhs_evaled = eval(rhs)
                return lhs_evaled-rhs_evaled
            end
        elseif op == :<
            function (x)
                for (key, value) in union(x, constants, sets)
                    eval(Meta.parse("$key = $value"))
                end
                lhs_evaled = eval(lhs)
                rhs_evaled = eval(rhs)
                return rhs_evaled-lhs_evaled
            end
        else
            error("unexpected operator ", op)
        end
    end
    return constr_fn
end


function find_vars_in_eq(model, eq, variables)
    """Finds and returns all varkeys in equation. """
    lhs, rhs, op = eq.args[1], eq.args[2], GAMSFiles.eqops[GAMSFiles.getname(eq)]
    vks = []
    for (var, vinfo) in variables
        if var isa GAMSFiles.GText
            if GAMSFiles.hasvar(lhs, var.text) || GAMSFiles.hasvar(rhs, var.text)
                push!(vks, var)
            end
        elseif var isa GAMSFiles.GArray
            if GAMSFiles.hasvar(lhs, var.name) || GAMSFiles.hasvar(rhs, var.name)
                push!(vks, var)
            end
        end
    end
    vars = []
    for vk in vks
        if vk isa GAMSFiles.GArray
            push!(vars, JuMP.variable_by_name(model, string(vk.name)))
        elseif vk isa GAMSFiles.GText
            push!(vars, JuMP.variable_by_name(model, string(vk.text)))
        end
    end
    @assert length(vars) == length(vks)
    return vars
end

function generate_variables!(model, gams)
    """Takes gams["variables"] and turns them into JuMP.Variables"""
    gams_variables = gams["variables"]
    for (v, vinfo) in gams_variables
        idxs = nothing
        if v isa GAMSFiles.GArray
            sym = String(v.name)
            axs = vinfo.axs
            idxs = collect(Base.product([collect(ax) for ax in axs]...))
            if idxs[1] isa Tuple{Int64}
                var = @variable(model, [1:length(idxs)], base_name = sym)
            else
                var = @variable(model, [size(idxs)], base_name = sym)
            end
        elseif v isa GAMSFiles.GText
            sym = String(v.text)
            @variable(model, base_name = sym)
        end
        for (prop, val) in vinfo.assignments
            inds = map(x->x.val, prop.indices)
            if isa(prop, Union{GAMSFiles.GText, GAMSFiles.GArray})
                c = val.val
#                 if GAMSFiles.getname(prop) ∈ ("l", "fx")
#                     x0[Symbol(v.name, inds[1])] = c
                if GAMSFiles.getname(prop) == "lo"
                    set_lower_bound(var[inds...], c)
                elseif GAMSFiles.getname(prop) == "up"
                    set_upper_bound(var[inds...], c)
                end
            end
        end
    end
    return JuMP.all_variables(model)
end

# Initialize JuMP.Model
model = Model(Gurobi.Optimizer)

# Parsing GAMS Files
lexed = GAMSFiles.lex(filename)
gams = GAMSFiles.parsegams(filename)
GAMSFiles.parseconsts!(gams)

vars = GAMSFiles.getvars(gams["variables"])
sets = gams["sets"]
preexprs, bodyexprs = Expr[], Expr[]
if haskey(gams, "parameters") && haskey(gams, "assignments")
    GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
end

# Getting variables
constants_to_global(gams)
all_vars = generate_variables!(model, gams)

# Getting objective
# @objective(model, Min, sum(JuMP.variable_by_name(model, string(i)) for i in gams["minimizing"]))


# # Creating BlackBoxFunction expressions
gm = GlobalModel(model = model)
consts = constants(gams)
for (key, eq) in gams["equations"]
    if key isa GAMSFiles.GText
        constr_fn = eq_to_fn(eq, sets, consts)
        vars = find_vars_in_eq(model, eq, gams["variables"])
        add_nonlinear_constraint(gm, constr_fn, vars,
                                 name = GAMSFiles.getname(key))
       add_fn!(model, bbf)
    elseif key isa GAMSFiles.GArray
        axes = GAMSFiles.getaxes(key.indices, sets)
        ar = sets_to_idxs(key.indices, sets)
        for allocation in ar
            next_all = Dict(key.indices[i].text => allocation[i] for i=1:length(key.indices))
            new_key = string(key.name, "_", allocation)
            new_set = merge(sets, next_all)
            constr_fn = eq_to_fn(eq, new_set, consts)
            vars = find_vars_in_eq(model, eq, gams["variables"])
            add_nonlinear_constraint(gm, constr_fn, vars,
                                     name = new_key)
        end
    end
end
#
# inp = Dict(vk => 1 for vk in model.vks)
# model.bbfs[2](inp)
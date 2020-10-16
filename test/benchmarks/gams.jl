#=
gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-10
=#
include("../load.jl")
include("../../../GAMSFiles.jl/src/GAMSFiles.jl")
using .GAMSFiles
using IterTools

filename = "data/baron_nc_ns/problem3.13.gms"

function vars_to_vks(vars)
    """ Turns gams["variables"] to varkeys for use in GlobalModel. """
    vks = Symbol[];
    vartypes = Dict()
    for (v, vinfo) in vars
        if isa(v, GAMSFiles.GText)
            vk = Symbol(v.text)
            push!(vks, vk);
            vartypes[vk] = vinfo.typ;
        elseif isa(v, GAMSFiles.GArray)
            basekey = v.name
            axs = vinfo.axs
            idx = [i for a in axs for i in a]
            for i in idx
                push!(vks, Symbol(v.name, i))
                vartypes[Symbol(v.name, i)] = vinfo.typ
            end
        end
    end
    @assert length(vks) == length(vartypes)
    return vks, vartypes
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


function find_vks_in_eq(eq, variables)
    """Finds and returns all varkeys in equation. """
    lhs, rhs, op = eq.args[1], eq.args[2], GAMSFiles.eqops[GAMSFiles.getname(eq)]
    vks = Symbol[];
    vars = []
    for (var, vinfo) in variables
        if var isa GAMSFiles.GText
            if GAMSFiles.hasvar(lhs, var.text) || GAMSFiles.hasvar(rhs, var.text)
                push!(vars, var)
            end
        elseif var isa GAMSFiles.GArray
            if GAMSFiles.hasvar(lhs, var.name) || GAMSFiles.hasvar(rhs, var.name)
                push!(vars, var)
            end
        end
    end
    vks, _ = vars_to_vks(Dict((var => gams["variables"][var]) for var in vars))
    return vks
end

function get_constants(gams::Dict{String, Any})
    """Returns all constants from GAMS Model. """
    constants = Dict(); #
    for p in ("parameters", "tables")
        if haskey(gams, p)
            for (k, v) in gams[p]
                v = v isa Ref ? v[] : v
                constants[Symbol(GAMSFiles.getname(k))] = v;
            end
        end
    end
    return constants
end

function get_bounds(variables)
    """Returns all bounds and initial points from gams["variables"]."""
    lbs = Dict(); ubs = Dict(); x0 = Dict();
    for (v, vinfo) in variables
        for (prop, val) in vinfo.assignments
            inds = nothing
            if isa(prop, GAMSFiles.GText)
                inds = (i,)
            elseif isa(prop, GAMSFiles.GArray)
                inds = map(x->x.val, prop.indices)
            end
            if isa(prop, Union{GAMSFiles.GText, GAMSFiles.GArray})
                c = val.val
                if GAMSFiles.getname(prop) ∈ ("l", "fx")
                    x0[Symbol(v.name, inds[1])] = c
                elseif GAMSFiles.getname(prop) == "lo"
                    lbs[Symbol(v.name, inds[1])] = c
                elseif GAMSFiles.getname(prop) == "up"
                    ubs[Symbol(v.name, inds[1])] = c
                end
            end
        end
    end
    return lbs, ubs, x0
end

function sets_to_idxs(setnames, sets)
    axs = GAMSFiles.getaxes(setnames, sets)
    if axs isa Tuple{}
        return nothing
    elseif axs isa Tuple{Base.OneTo{Int},Vararg{Base.OneTo{Int}}}
        ar = collect(Base.product([collect(ax) for ax in axs]...))
        return ar
    else
        return OffsetArray{Float64}(undef, axs)
    end
end

# Parsing GAMS Files
lexed = GAMSFiles.lex(filename)
gams = GAMSFiles.parsegams(filename)
GAMSFiles.parseconsts!(gams)

constants = get_constants(gams)

vars = GAMSFiles.getvars(gams["variables"])
sets = gams["sets"]
preexprs, bodyexprs = Expr[], Expr[]
if haskey(gams, "parameters") && haskey(gams, "assignments")
    GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
end

# Variable data parsing
vks, vartypes = vars_to_vks(gams["variables"]);
lbs, ubs, x0 = get_bounds(gams["variables"]);

# Initialize GlobalModel
all_vars = GAMSFiles.allvars(gams)
c = zeros(length(vks));
c[findall(i -> i == Symbol(gams["minimizing"]), vks)] .= 1;
model = OCT.GlobalModel(c = c, vks = vks);

# Create BlackBoxFunction expressions
for (key, eq) in gams["equations"]
    if key isa GAMSFiles.GText
        constr_fn = eq_to_fn(eq, sets, constants)
        bbf = OCT.BlackBoxFunction(fn = constr_fn, vks = find_vks_in_eq(eq, gams["variables"]),
                               name = GAMSFiles.getname(key))
       add_fn!(model, bbf)
    elseif key isa GAMSFiles.GArray
        axes = GAMSFiles.getaxes(key.indices, sets)
        ar = sets_to_idxs(key.indices, sets)
        for allocation in ar
            next_all = Dict(key.indices[i].text => allocation[i] for i=1:length(key.indices))
            new_key = string(key.name, "_", allocation)
            new_set = merge(sets, next_all)
            constr_fn = eq_to_fn(eq, new_set, constants)
            bbf = OCT.BlackBoxFunction(fn = constr_fn, vks = find_vks_in_eq(eq, gams["variables"]),
                                    name = new_key)
            add_fn!(model, bbf)
        end
    end
end

inp = Dict(vk => 1 for vk in model.vks)
model.bbfs[2](inp)
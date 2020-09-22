#=
gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-10
=#
include("../load.jl")
include("../../../GAMSFiles.jl/src/GAMSFiles.jl")
using .GAMSFiles
const GF = GAMSFiles

filename = "data/baron_nc_ns/problem3.13.gms"

function vars_to_vks(vars)
    """ Turns gams["variables"] to varkeys for use in ModelData. """
    vks = Symbol[];
    vartypes = Dict()
    for (v, vinfo) in vars
        if isa(v, GF.GText)
            vk = Symbol(v.text)
            push!(vks, vk);
            vartypes[vk] = vinfo.typ;
        elseif isa(v, GF.GArray)
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

function find_vks_in_eq(eq, variables)
    """Finds and returns all varkeys in equation. """
    lhs, rhs, op = eq.args[1], eq.args[2], GF.eqops[GF.getname(eq)]
    vks = Symbol[];
    vars = []
    for (var, vinfo) in variables
        if var isa GF.GText
            if GF.hasvar(lhs, var.text) || GF.hasvar(rhs, var.text)
                push!(vars, var)
            end
        elseif var isa GF.GArray
            if GF.hasvar(lhs, var.name) || GF.hasvar(rhs, var.name)
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
                constants[Symbol(GF.getname(k))] = v;
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
            if isa(prop, GF.GText)
                inds = (i,)
            elseif isa(prop, GF.GArray)
                inds = map(x->x.val, prop.indices)
            end
            if isa(prop, Union{GF.GText, GF.GArray})
                c = val.val
                if GF.getname(prop) ∈ ("l", "fx")
                    x0[Symbol(v.name, inds[1])] = c
                elseif GF.getname(prop) == "lo"
                    lbs[Symbol(v.name, inds[1])] = c
                elseif GF.getname(prop) == "up"
                    ubs[Symbol(v.name, inds[1])] = c
                end
            end
        end
    end
    return lbs, ubs, x0
end

# Parsing GAMS Files
lexed = GF.lex(filename)
gams = GF.parsegams(filename)
GF.parseconsts!(gams)

constants = get_constants(gams)

# Variable data parsing
vks, vartypes = vars_to_vks(gams["variables"]);
lbs, ubs, x0 = get_bounds(gams["variables"]);

# Initialize ModelData
all_vars = GF.allvars(gams)
c = zeros(length(vks));
c[findall(i -> i == Symbol(gams["minimizing"]), vks)] .= 1;
model = OCT.ModelData(c = c, vks = vks);

# Create BlackBoxFunction expressions
for (key, eq) in gams["equations"]
    @assert(eq isa GF.GCall && length(eq.args)==2)
    lhs, rhs, op = eq.args[1], eq.args[2], GF.eqops[GF.getname(eq)]
    lhsexpr = convert(Expr, lhs)
    rhsexpr = convert(Expr, rhs)
    GF.replace_reductions!(lhsexpr, gams["sets"])
    GF.replace_reductions!(rhsexpr, gams["sets"])
    constr_fn = let lhs = lhsexpr, rhs = rhsexpr, op = op, constants = constants
        if op in [:(=), :>]
            function (x)
                for (key, value) in union(x, constants, gams["sets"])
                    eval(Meta.parse("$key = $value"))
                end
                lhs_evaled = eval(lhs)
                rhs_evaled = eval(rhs)
                return lhs_evaled-rhs_evaled
            end
        elseif op == :<
            function (x)
                for (key, value) in union(x, constants, gams["sets"])
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
    # Getting vks in the equation.
    bbf = OCT.BlackBoxFunction(fn = constr_fn, vks = find_vks_in_eq(eq, gams["variables"]),
                               name = GF.getname(key))
    add_fn!(model, bbf)
end

inp = Dict(vk => 1 for vk in model.vks)
model.fns[2](inp)
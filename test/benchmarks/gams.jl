#=
gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-10
=#

include("../../../GAMSFiles.jl/src/GAMSFiles.jl")
using .GAMSFiles

"""Returns all constants from GAMS Model. """
function constants(gams::Dict{String, Any})
    consts = Dict(); #
    for p in ("parameters", "tables")
        if haskey(gams, p)
            for (k, v) in gams[p]
                v = v isa Ref ? v[] : v
                consts[GAMSFiles.getname(k)] = v;
            end
        end
    end
    return consts
end

""" Moves GAMS constants into the global scope.
    NOTE: This is only for debugging purposes, could cause bugs.
"""
function constants_to_global(gams::Dict{String, Any})
    consts = constants(gams)
    for (key, value) in consts
        eval(Meta.parse("$(key) = $(value)"))
    end
end

""" Turns GAMSFiles.GCall into an Expr. """
function eq_to_expr(eq::GAMSFiles.GCall, sets::Dict{String, Any})
    @assert(length(eq.args)==2)
    lhs, rhs, op = eq.args[1], eq.args[2], GAMSFiles.eqops[GAMSFiles.getname(eq)]
    lhsexpr = convert(Expr, lhs)
    rhsexpr = convert(Expr, rhs)
    GAMSFiles.replace_reductions!(lhsexpr, sets)
    GAMSFiles.replace_reductions!(rhsexpr, sets)
    if op in [:(=), :>]
        :($(lhsexpr) - $(rhsexpr))
    elseif op == :<
        :($(rhsexpr) - $(lhsexpr))
    else
        error("unexpected operator ", op)
    end
end

""" Returns whether a GAMSFiles.GCall is an equality. """
function is_equality(eq::GAMSFiles.GCall)
    @assert(length(eq.args)==2)
    GAMSFiles.eqops[GAMSFiles.getname(eq)] == :(=)
end


function find_vars_in_eq(eq::GAMSFiles.GCall, vardict::Dict{String, Any})
    """Finds and returns all varkeys in equation. """
    vars = String[]
    lhs, rhs = eq.args[1], eq.args[2]
    for (var, vinfo) in vardict
        if GAMSFiles.hasvar(lhs, var) || GAMSFiles.hasvar(rhs, var)
            push!(vars, var)
        end
    end
    return vars
end

"""Takes gams and turns them into JuMP.Variables"""
function generate_variables!(model::JuMP.Model, gams::Dict{String, Any})
    gamsvars = GAMSFiles.allvars(gams)
    vardict = Dict{String, Any}()
    constdict = Dict{String, Any}()
    for (var, vinfo) in gamsvars
        if vinfo isa Array
            sizes = [Base.OneTo(vin) for vin in size(vinfo)]
            dims = length(sizes)
            nv = JuMP.Containers.DenseAxisArray{JuMP.VariableRef}(undef, sizes...)
            for idx in eachindex(nv)
                nv[idx] = @variable(model)
                set_name(nv[idx], "$(var)[$(join(Tuple(idx),","))]")
                fix(nv[idx], vinfo[idx])
            end
            constdict[var] = nv
        elseif vinfo isa Real
            nv = @variable(model, base_name = var)
            model[Symbol(var)] = nv
            fix(nv, vinfo)
            constdict[var] = nv
        else
            axs = vinfo.axs
            nv = nothing
            if axs == ()
                nv = @variable(model, base_name = var)
                model[Symbol(var)] = nv
            else
                nv = JuMP.Containers.DenseAxisArray{JuMP.VariableRef}(undef, axs...)
                for idx in eachindex(nv)
                    nv[idx] = @variable(model)
                    set_name(nv[idx], "$(var)[$(join(Tuple(idx),","))]")
                end
                model[Symbol(var)] = nv
            end
            vardict[var] = nv
            for (prop, val) in vinfo.assignments
                inds = map(x->x.val, prop.indices)
                if isa(prop, Union{GAMSFiles.GText, GAMSFiles.GArray})
                    c = val.val
    #                 if GAMSFiles.getname(prop) ∈ ("l", "fx")
    #                     x0[Symbol(nv.name, inds[1])] = c
                    if GAMSFiles.getname(prop) == "lo"
                        set_lower_bound(nv[inds...], c)
                    elseif GAMSFiles.getname(prop) == "up"
                        set_upper_bound(nv[inds...], c)
                    end
                end
            end
        end
    end
    return vardict, constdict
end

""" Converts a GAMS optimization model to a GlobalModel."""
function GAMS_to_GlobalModel(filename::String)
    model = JuMP.Model(Gurobi.Optimizer)
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
    vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables
    consts = constants(gams)                              # Dict of values of constants.

    # Getting objective
    if gams["minimizing"] isa String
        @objective(model, Min, JuMP.variable_by_name(model, gams["minimizing"]))
    else
        @objective(model, Min, sum([JuMP.variable_by_name(i) for i in gams["minimizing"]]))
    end

    # Creating GlobalModel
    gm = GlobalModel(model = model)
    equations = [] # For debugging purposes...
    for (key, eq) in gams["equations"]
        push!(equations, key => eq)
    end
    for (key, eq) in equations
        if key isa GAMSFiles.GText
            constr_expr = eq_to_expr(eq, sets)
            # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(Symbol(constkey) => constdict[constkey] for constkey in constkeys)
            for (constkey, constvar) in const_pairs
                constr_expr = substitute(constr_expr, :($constkey) => constvar)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
            input = Symbol.(varkeys)
            constr_fn = :(($(input...),) -> $(constr_expr))
            if length(input) == 1
                constr_fn = :($(input...) -> $(constr_expr))
            end
            add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                                     equality = is_equality(eq), name = GAMSFiles.getname(key))
        elseif key isa GAMSFiles.GArray
            axs = GAMSFiles.getaxes(key.indices, sets)
            idxs = collect(Base.product([collect(ax) for ax in axs]...))
            names = [key.name * string(idx) for idx in idxs]
            constr_expr = eq_to_expr(eq, sets)
                    # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(Symbol(constkey) => constdict[constkey] for constkey in constkeys)
            for (constkey, constvar) in const_pairs
                constr_expr = substitute(constr_expr, :($constkey) => constvar)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
            input = Symbol.(varkeys)
            constr_fn = :(($(input...),) -> $(constr_expr))
            if length(input) == 1
                constr_fn = :($(input...) -> $(constr_expr))
            end
            constr_fns = []
            for idx in idxs
                new_fn = copy(constr_fn)
                for ax_number in 1:length(axs)
                    new_fn = substitute(new_fn, Symbol(key.indices[ax_number].text) => idx[ax_number]);
                end
                push!(constr_fns, new_fn)
            end
            for i = 1:length(idxs)
                add_nonlinear_or_compatible(gm, constr_fns[i], vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                                         equality = is_equality(eq),
                                         name = names[i])
            end
        end
    end
    return gm
end

filename = "data/baron_nc_ns/problem3.13.gms"
gm = GAMS_to_GlobalModel(filename)
@test length(gm.vars) == 103
@test length(gm.bbfs) == 13


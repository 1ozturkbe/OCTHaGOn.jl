using GAMSFiles

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

""" Finds and returns all varkeys in equation. """
function find_vars_in_eq(eq::GAMSFiles.GCall, vardict::Dict{Symbol, Any})
    vars = Symbol[]
    lhs, rhs = eq.args[1], eq.args[2]
    for (var, vinfo) in vardict
        if GAMSFiles.hasvar(lhs, string(var)) || GAMSFiles.hasvar(rhs, string(var))
            push!(vars, var)
        end
    end
    return vars
end

""" Takes gams and turns them into JuMP.Variables"""
function generate_variables!(model::JuMP.Model, gams::Dict{String, Any})
    gamsvars = GAMSFiles.allvars(gams)
    vardict = Dict{Symbol, Any}()
    constdict = Dict{Symbol, Any}()
    for (var, vinfo) in gamsvars
        if vinfo isa Union{Array, Real}
            model[Symbol(var)] = vinfo
            constdict[Symbol(var)] = vinfo
        elseif vinfo isa Base.RefValue
            model[Symbol(var)] = vinfo.x
            constdict[Symbol(var)] = vinfo.x
        else
            axs = vinfo.axs
            if axs == ()
                nv = @variable(model, base_name = var)
                model[Symbol(var)] = nv
                vardict[Symbol(var)] = nv
            else
                nv = JuMP.Containers.DenseAxisArray{JuMP.VariableRef}(undef, axs...)
                for idx in eachindex(nv)
                    nv[idx] = @variable(model)
                    set_name(nv[idx], "$(var)[$(join(Tuple(idx),","))]")
                end
                model[Symbol(var)] = nv
                vardict[Symbol(var)] = nv
            end
            if vinfo.typ != "free"
                if vinfo.typ == "positive"
                    JuMP.set_lower_bound.(model[Symbol(var)], 0)
                elseif vinfo.typ == "negative"
                    JuMP.set_upper_bound.(model[Symbol(var)], 0)
                elseif vinfo.typ == "binary"
                    JuMP.set_binary.(model[Symbol(var)])
                elseif vinfo.typ == "integer"
                    JuMP.set_integer.(model[Symbol(var)])
                else
                    throw(OCTException("Type $(vinfo.typ) unknown for variable $(var)."))
                end
            end
            for (prop, val) in vinfo.assignments
                if hasproperty(prop, :indices)
                    inds = map(x->x.val, prop.indices)
                    nv = model[Symbol(var)][inds...]
                    c = val.val
                    if prop.name ∈ ("l", "fx")
                        JuMP.set_start_value(nv, c)
                    elseif prop.name == "lo"
                        JuMP.set_lower_bound(nv, c)
                    elseif prop.name == "up"
                        JuMP.set_upper_bound(nv, c)
                    end
                else
                    nv = model[Symbol(var)]
                    c = val.val
                    if prop.text ∈ ("l", "fx")
                        JuMP.set_start_value(nv, c)
                    elseif prop.text == "lo"
                        JuMP.set_lower_bound(nv, c)
                    elseif prop.text == "up"
                        JuMP.set_upper_bound(nv, c)
                    end
                end
            end
        end
    end
    return vardict, constdict
end

""" Converts a GAMS optimization model to a GlobalModel."""
function GAMS_to_GlobalModel(GAMS_DIR::String, filename::String)
    model = JuMP.Model()
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(GAMS_DIR * filename)
    gams = GAMSFiles.parsegams(GAMS_DIR * filename)
    GAMSFiles.parseconsts!(gams)

    vars = GAMSFiles.getvars(gams["variables"])
    sets = Dict{String, Any}()
    if haskey(gams, "sets")
        sets = gams["sets"]
    end
    preexprs, bodyexprs = Expr[], Expr[]
    if haskey(gams, "parameters") && haskey(gams, "assignments")
        GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
    end

    # Getting variables
    vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables

    # Getting objective
    if gams["minimizing"] isa String
        @objective(model, Min, JuMP.variable_by_name(model, gams["minimizing"]))
    else
        @objective(model, Min, sum([JuMP.variable_by_name(i) for i in gams["minimizing"]]))
    end

    # Creating GlobalModel
    gm = GlobalModel(model = model, name = filename)
    equations = [] # For debugging purposes...
    for (key, eq) in gams["equations"]
        push!(equations, key => eq)
    end
    for (key, eq) in equations
        if key isa GAMSFiles.GText
            constr_expr = eq_to_expr(eq, sets)
            # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(constkey => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = substitute(constr_expr, :($constkey) => constval)
            end
            # Designate free variables
            varkeys = find_vars_in_eq(eq, vardict)
            if !(Symbol(gams["minimizing"]) in varkeys)
                vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> $(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> $(constr_expr))
                end
                add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                                        equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key))
            else
                constr_expr = substitute(constr_expr, :($(Symbol(gams["minimizing"]))) => 0)
                # ASSUMPTION: objvar has positive coefficient, and is on the greater size. 
                op = GAMSFiles.eqops[GAMSFiles.getname(eq)]
                if !(op in [:<, :>])
                    throw(OCTException("Please make sure GAMS model has objvar on the greater than size of inequalities, " *
                                        " with a leading coefficient of 1."))
                end
                varkeys = filter!(x -> x != Symbol(gams["minimizing"]), varkeys)
                vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
                input = Symbol.(varkeys)
                constr_fn = :(($(input...),) -> -$(constr_expr))
                if length(input) == 1
                    constr_fn = :($(input...) -> -$(constr_expr))
                end
                add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
                    dependent_var = vardict[Symbol(gams["minimizing"])], equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key))
            end
        elseif key isa GAMSFiles.GArray
            axs = GAMSFiles.getaxes(key.indices, sets)
            idxs = collect(Base.product([collect(ax) for ax in axs]...))
            names = [gm.name * "_" * key.name * string(idx) for idx in idxs]
            constr_expr = eq_to_expr(eq, sets)
                    # Substitute constant variables
            constkeys = find_vars_in_eq(eq, constdict)
            const_pairs = Dict(Symbol(constkey) => model[constkey] for constkey in constkeys)
            for (constkey, constval) in const_pairs
                constr_expr = substitute(constr_expr, :($constkey) => constval)
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

# """ Solver wrapper for GAMS benchmarking. """
# function nonlinear_solve(gm::GlobalModel; solver = IPOPT_SILENT)
#     nonlinearize!(gm)
#     set_optimizer(gm, solver)
#     optimize!(gm)
# end
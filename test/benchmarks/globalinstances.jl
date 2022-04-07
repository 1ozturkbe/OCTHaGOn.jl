dir = "../OCTHaGOn_benchmarks/gams/global/"
filenames = readdir(dir)
GAMS_DIR = dir

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/gams.jl")

valid_filenames = []

pd = DataFrame()
n_max = 100
pd = CSV.read(dir * "problem_stats.csv", DataFrame)
pd = pd[pd.all_bounded .>= 0.5, :]
pd = pd[pd.n_vars .<= n_max, :]

for filename in pd.name
    filename = filename * ".gms"
    @info "Trying " * filename * "."
    model = JuMP.Model()
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(GAMS_DIR * filename)
    gams = []
    try
        gams = GAMSFiles.parsegams(GAMS_DIR * filename)
    catch e
        if e isa KeyError
            @warn filename * " failed due to KeyError."
        elseif e isa LoadError
            @warn filename * " failed due to an InvalidCharError. "
        else
            @warn filename * " failed due to an unknown Error."
        end
        continue
    end
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
    if length(vardict) <= n_max
        @info filename * " added with $(length(vardict)) variables."
        push!(valid_filenames, filename)
    end
end

@assert length(valid_filenames) == length(pd.name)

# gms = Dict()
# for filename in filenames
#     try
#         @info "Trying " * filename * "."
#         gm = GAMS_to_GlobalModel(dir, filename)
# #             println("    Problem NL constraints: " * string(length(gm.bbls)))
#         # types = JuMP.list_of_constraint_types(gm.model)
#         # if !isempty(types)
#         #     total_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
# #                 println("    Problem total constraints: " * string(total_constraints))
#         # end
#         merge!(gms, gm)
#     catch
#         # throw(OCTHaGOnException(filename * " has an import issue."))
#     end
# end

# allconstrs = []
# for (key, values) in gms
#     push!(allconstrs, values...)
# end

# filename = "4stufen.gms"
# GAMS_DIR = dir

# model = JuMP.Model()
# # Parsing GAMS Files
# lexed = GAMSFiles.lex(GAMS_DIR * filename)
# gams = GAMSFiles.parsegams(GAMS_DIR * filename)
# GAMSFiles.parseconsts!(gams)

# vars = GAMSFiles.getvars(gams["variables"])
# sets = Dict{String, Any}()
# if haskey(gams, "sets")
#     sets = gams["sets"]
# end
# preexprs, bodyexprs = Expr[], Expr[]
# if haskey(gams, "parameters") && haskey(gams, "assignments")
#     GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
# end

# # Getting variables
# vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables

# # Getting objective
# if gams["minimizing"] isa String
#     @objective(model, Min, JuMP.variable_by_name(model, gams["minimizing"]))
# else
#     @objective(model, Min, sum([JuMP.variable_by_name(i) for i in gams["minimizing"]]))
# end

# # Creating GlobalModel
# gm = GlobalModel(model = model, name = filename)
# equations = [] # For debugging purposes...
# for (key, eq) in gams["equations"]
#     push!(equations, key => eq)
# end
# for (key, eq) in equations
#     if key isa GAMSFiles.GText
#         constr_expr = eq_to_expr(eq, sets)
#         # Substitute constant variables
#         constkeys = find_vars_in_eq(eq, constdict)
#         const_pairs = Dict(constkey => model[constkey] for constkey in constkeys)
#         for (constkey, constval) in const_pairs
#             constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
#         end
#         # Designate free variables
#         varkeys = find_vars_in_eq(eq, vardict)
#         if !(Symbol(gams["minimizing"]) in varkeys)
#             vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
#             input = Symbol.(varkeys)
#             constr_fn = :(($(input...),) -> $(constr_expr))
#             if length(input) == 1
#                 constr_fn = :($(input...) -> $(constr_expr))
#             end
#             add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
#                                     equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key))
#         else
#             constr_expr = OCTHaGOn.substitute(constr_expr, :($(Symbol(gams["minimizing"]))) => 0)
#             # ASSUMPTION: objvar has positive coefficient, and is on the greater size. 
#             op = GAMSFiles.eqops[GAMSFiles.getname(eq)]
#             if !(op in [:<, :>])
#                 throw(OCTHaGOnException("Please make sure GAMS model has objvar on the greater than size of inequalities, " *
#                                     " with a leading coefficient of 1."))
#             end
#             varkeys = filter!(x -> x != Symbol(gams["minimizing"]), varkeys)
#             vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
#             input = Symbol.(varkeys)
#             constr_fn = :(($(input...),) -> -$(constr_expr))
#             if length(input) == 1
#                 constr_fn = :($(input...) -> -$(constr_expr))
#             end
#             add_nonlinear_or_compatible(gm, constr_fn, vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
#                 dependent_var = vardict[Symbol(gams["minimizing"])], equality = is_equality(eq), name = gm.name * "_" * GAMSFiles.getname(key))
#         end
#     elseif key isa GAMSFiles.GArray
#         axs = GAMSFiles.getaxes(key.indices, sets)
#         idxs = collect(Base.product([collect(ax) for ax in axs]...))
#         names = [gm.name * "_" * key.name * string(idx) for idx in idxs]
#         constr_expr = eq_to_expr(eq, sets)
#         # Substitute constant variables
#         constkeys = find_vars_in_eq(eq, constdict)
#         const_pairs = Dict(Symbol(constkey) => model[constkey] for constkey in constkeys)
#         for (constkey, constval) in const_pairs
#             constr_expr = OCTHaGOn.substitute(constr_expr, :($constkey) => constval)
#         end
#         # Designate free variables
#         varkeys = find_vars_in_eq(eq, vardict)
#         vars = Array{VariableRef}(flat([vardict[varkey] for varkey in varkeys]))
#         input = Symbol.(varkeys)
#         constr_fn = :(($(input...),) -> $(constr_expr))
#         if length(input) == 1
#             constr_fn = :($(input...) -> $(constr_expr))
#         end
#         constr_fns = []
#         for idx in idxs
#             new_fn = copy(constr_fn)
#             for ax_number in 1:length(axs)
#                 new_fn = OCTHaGOn.substitute(new_fn, Symbol(key.indices[ax_number].text) => idx[ax_number]);
#             end
#             push!(constr_fns, new_fn)
#         end
#         for i = 1:length(idxs)
#             add_nonlinear_or_compatible(gm, constr_fns[i], vars = vars, expr_vars = [vardict[varkey] for varkey in varkeys],
#                                      equality = is_equality(eq),
#                                      name = names[i])
#         end
#     end
# end
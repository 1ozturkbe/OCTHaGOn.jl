function rocket100()
    model = JuMP.Model()
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(OCTHaGOn.GAMS_DIR * "rocket100.gms")
    gams = GAMSFiles.parsegams(OCTHaGOn.GAMS_DIR * "rocket100.gms")
    GAMSFiles.parseconsts!(gams)

    vars = GAMSFiles.getvars(gams["variables"])
    sets = Dict{String, Any}()
    preexprs, bodyexprs = Expr[], Expr[]

    # Getting variables
    vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables
    objvar = model[:objvar]
    x = [objvar] # :x1 is the objective variable
    symbs = [Symbol("x$(i)") for i=2:608]
    append!(x, [model[symb] for symb in symbs])
    model[:x] = x
    @objective(model, Min, objvar)
    
    # Initializing GM
    gm = GlobalModel(model = model, name = "rocket100.gms")


    # (Re)writing constraints
    add_nonlinear_constraint(gm, :(x -> -310*sqrt(x[3])*exp(500 - 500*x[104]) + x[508]), 
                             name = "e1", vars = [x[3], x[104], x[508]], equality = true)
    for i=1:100
        add_linked_constraint(gm, gm.bbls[end], [x[3+i], x[104+i], x[508+i]])
    end

    add_nonlinear_constraint(gm, :(x -> -sqrt(1/x[104]) + x[205]), 
                             name = "e102", vars = [x[104], x[205]], equality = true)
    for i=1:100
        add_linked_constraint(gm, gm.bbls[end], [x[104+i], x[205+i]])
    end

    @constraint(gm.model, -objvar >= x[204])
    @objective(gm.model, Min, objvar)

    add_nonlinear_constraint(gm, :(x -> -0.5*x[2]*(x[3] + x[4]) - x[104] + x[105]),
                             name = "e204", vars = [x[2], x[3], x[4], x[104], x[105]], equality = true)
    for i=1:99
        add_linked_constraint(gm, gm.bbls[end], [x[2], x[3+i], x[4+i], x[104+i], x[105+i]])
    end

    add_nonlinear_constraint(gm, :(x -> -0.5*((x[408] - x[307]*x[206] - x[509])/x[307] + (x[407] - x[306]*x[205] - x[508])/x[306]) *
                                   x[2] - x[3] + x[4]),
                                   name = "e304", vars = [x[2], x[3], x[4], x[205], x[206], x[306], 
                                                          x[307], x[407], x[408], x[508], x[509]], equality = true)

    for i=1:99
        add_linked_constraint(gm, gm.bbls[end], [x[2], x[3+i], x[4+i], x[205+i], x[206+i], x[306+i], 
                                                 x[307+i], x[407+i], x[408+i], x[508+i], x[509+i]]) 
    end

    add_nonlinear_constraint(gm, :(x -> x[2]*(x[407] + x[408]) - x[306] + x[307]), 
                                   name = "e404", vars = [x[2], x[306], x[307], x[407], x[408]], equality = true)

    for i=1:99
        add_linked_constraint(gm, gm.bbls[end], [x[2], x[306+i], x[307+i], x[407+i], x[408+i]])
    end

    return gm
end

using Ipopt
set_optimizer(gm.model, Ipopt.Optimizer)
nonlinearize!(gm)
optimize!(gm)
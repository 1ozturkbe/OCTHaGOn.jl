function rocket100()
    model = JuMP.Model()
    # Parsing GAMS Files
    lexed = GAMSFiles.lex(OCT.GAMS_DIR * "\\gms\\rocket100.gms")
    gams = GAMSFiles.parsegams(OCT.GAMS_DIR * "\\gms\\rocket100.gms")
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
    
    # Initializing GM
    gm = GlobalModel(model = model, name = filename)


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

    @constraint(gm.model, - objvar >= x[204])
    @objective(gm.model, Min, objvar)

    return gm
end
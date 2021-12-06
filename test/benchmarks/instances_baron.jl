file = open(OCT.BARON_DIR*"instances.txt", "r")
gams_problem_names = []
while !eof(file) 
    # read a new / next line for every iteration          
    s = readline(file)         
    push!(gams_problem_names, string(s) * ".gms")
 end

 for name in gams_problem_names
    try
        gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, name)
        @info "$(name) worked."
    catch
        @info "$(name) didn't work."
    end
 end

 name = "flay05m.gms" # GOOD EXAMPLE
 gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, name)
 set_optimizer(gm, CPLEX_SILENT)
 globalsolve_and_time!(gm)
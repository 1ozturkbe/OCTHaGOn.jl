file = open(OCT.BARON_DIR*"instances.txt", "r")
gams_problem_names = []
while !eof(file) 
    # read a new / next line for every iteration          
    s = readline(file)         
    push!(gams_problem_names, string(s) * ".gms")
 end

 for name in gams_problem_names
    try
        gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
        @info "$(name) worked."
    catch
        @info "$(name) didn't work."
    end
 end

 name = "ex6_2_10.gms"
 name = "ex8_4_3.gms"
 name = "flay05m.gms" # GOOD EXAMPLE
 name = "pointpack08.gms" # GOOD EXAMPLE
 name = "ramsey.gms"
 gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
 globalsolve_and_time!(gm)

 add_relaxation_variables!(gm)
 relax_objective!(gm)
 add_tree_constraints!(gm)
 optimize!(gm)
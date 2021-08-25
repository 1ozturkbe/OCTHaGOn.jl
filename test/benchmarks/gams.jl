filename = "weapons.gms"
filename = "himmel16.gms"
filename = "ramsey.gms"
filename = "alan.gms"
filename = "gbd.gms"
filename = "hydro.gms"
filename = "pollut.gms"

gm = speed_reducer()
set_optimizer(gm, CPLEX_SILENT)
globalsolve_and_time!(gm)

name = "ex6_2_10.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

name = "ex7_2_3.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
set_param(gm, :step_penalty, 1e12)
globalsolve_and_time!(gm)

name = "ex8_4_3.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

name = "himmel16.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)
JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16
globalsolve_and_time!(gm)

name = "ramsey.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)
JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
globalsolve_and_time!(gm)

name = "pointpack08.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
bound!(gm.model, gm.model[:obj] => [-5, 5])
globalsolve_and_time!(gm)

name = "pollut.gms" # Large NL objective, but linear constraints
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

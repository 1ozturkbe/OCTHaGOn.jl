include("load.jl")
gm = nlp2(true)
set_optimizer(gm, SOLVER_SILENT)
globalsolve_and_time!(gm)
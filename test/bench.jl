include("load.jl")

gm = nlp2(true)
set_optimizer(gm, SOLVER_SILENT)
globalsolve_and_time!(gm)

using BARON
gm = GAMS_to_baron_model("../OCTHaGOn_benchmarks/gams/global/", "ex4_1_1.gms")
optimize!(gm)
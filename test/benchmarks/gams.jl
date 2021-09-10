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

name = "ex7_2_3.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
set_param(gm, :step_penalty, 1e12)
globalsolve_and_time!(gm)

name = "ex8_4_3.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

# LOGARITHM CAUSING ISSUES
name = "ramsey.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
globalsolve_and_time!(gm)

# Causing errors in hyperplane splits
name = "wastepaper4.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

# No idea yet? All o fthe ones below. 
name = "multiplants_mtg1a.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

# TAKES FOREVER...
name = "genpooling_meyer04.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)

# ALL SET TO GO 
name = "himmel16.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16

# ALL SET TO GO 
name = "kall_circles_c6b.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
time1 = time()
set_param(gm, :abstol, 1e-5)
uniform_sample_and_eval!(gm)
learn_constraint!(gm, max_depth=7)
globalsolve_and_time!(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET TO GO 
name = "pointpack08.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
bound!(gm.model, gm.model[:obj] => [-0.5, 0.5])
globalsolve_and_time!(gm)

# ALL SET TO GO 
name = "flay05m.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

# FIXING THIS ASAP, DUE TO THE WAY D IS COMPUTED. 
name = "fo9.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
globalsolve_and_time!(gm)

# ALL SET TO GO 
name = "o9_ar4_1.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
time1 = time()
set_param(gm, :abstol, 1e-5)
uniform_sample_and_eval!(gm)
learn_constraint!(gm, max_depth=7)
globalsolve_and_time!(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET, but no comparative solver
name = "sfacloc2_3_80.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR * "\\gms\\", name)
set_param(gm, :max_iterations, 500)
set_param(gm, :ignore_accuracy, true)
set_optimizer(gm, CPLEX_SILENT)
time1 = time()
uniform_sample_and_eval!(gm)
learn_constraint!(gm, max_depth=5)
while any(isempty(bbl.learners) for bbl in gm.bbls)
    for bbl in gm.bbls
        if isempty(bbl.learners)
            try
                learn_constraint!(bbl, max_depth=5)
            catch
                @info "BBL $(bbl.name) causing problems."
            end
        end
    end
end
add_tree_constraints!(gm)
optimize!(gm)
descend!(gm)
@info "Time elapsed: $(time()-time1)"
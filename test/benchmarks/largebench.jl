# filename = "weapons.gms"
# filename = "himmel16.gms"
# filename = "ramsey.gms"
# filename = "alan.gms"
# filename = "gbd.gms"
# filename = "hydro.gms"
# filename = "pollut.gms"

# gm = speed_reducer()
# set_optimizer(gm, CPLEX_SILENT)
# globalsolve_and_time!(gm)

# name = "ex7_2_3.gms"
# gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
# set_param(gm, :step_penalty, 1e12)
# set_optimizer(gm.model, CPLEX_SILENT)
# globalsolve_and_time!(gm)

# name = "ex8_4_3.gms"
# gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
# set_optimizer(gm.model, CPLEX_SILENT)
# globalsolve_and_time!(gm)

# # LOGARITHM CAUSING ISSUES
# name = "ramsey.gms"
# gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
# JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
# JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
# set_optimizer(gm.model, CPLEX_SILENT)
# globalsolve_and_time!(gm)

# # Causing errors in hyperplane splits
# name = "wastepaper4.gms"
# gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
# set_optimizer(gm.model, CPLEX_SILENT)
# globalsolve_and_time!(gm)

function optimize_with_more_depth(gm::GlobalModel)
    set_param(gm, :sample_coeff, 400)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm, max_depth = 8)
    globalsolve_and_time!(gm)
    return
end

# ALL SET TO GO 
name = "himmel16.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
set_optimizer(gm, CPLEX_SILENT)
JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16
time1 = time()
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET TO GO 
name = "kall_circles_c6b.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
set_optimizer(gm.model, CPLEX_SILENT)
set_param(gm, :step_penalty, 1e8)
set_param(gm, :equality_penalty, 1e8)
time1 = time()
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET TO GO 
name = "pointpack08.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
time1 = time()
set_optimizer(gm, CPLEX_SILENT)
bound!(gm.model, gm.model[:obj] => [0, 0.5])
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET TO GO 
name = "flay05m.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
set_optimizer(gm, CPLEX_SILENT)
time1 = time()
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"

# ALL SET TO GO
name = "fo9.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
set_optimizer(gm, CPLEX_SILENT)
time1 = time()
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"


# ALL SET TO GO 
name = "o9_ar4_1.gms"
gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
set_optimizer(gm.model, CPLEX_SILENT)
time1 = time()
optimize_with_more_depth(gm)
@info "Time elapsed: $(time()-time1)"

# # ALL SET, but no comparative solver
# name = "sfacloc2_3_80.gms"
# gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR, name)
# set_param(gm, :max_iterations, 500)
# set_param(gm, :ignore_accuracy, true)
# set_optimizer(gm, CPLEX_SILENT)
# time1 = time()
# uniform_sample_and_eval!(gm)
# learn_constraint!(gm, max_depth=5)
# while any(isempty(bbl.learners) for bbl in gm.bbls)
#     for bbl in gm.bbls
#         if isempty(bbl.learners)
#             try
#                 learn_constraint!(bbl, max_depth=5)
#             catch
#                 @info "BBL $(bbl.name) causing problems."
#             end
#         end
#     end
# end
# add_tree_constraints!(gm)
# optimize!(gm)
# descend!(gm)
# @info "Time elapsed: $(time()-time1)"
function test_survey_method(gm::GlobalModel = minlp(true))
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        update_vexity.(bbrs)  
    end
    surveysolve(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    bbc_idxs = [x isa BlackBoxClassifier for x in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while gm.cost[end] > gm.cost[end-1] .* (1 + get_param(gm, :tighttol)) && size(gm.solution_history, 1) <= 50
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end #TODO RESOLVE WHILE LOOP OVERRUN. 
    return gm
end

filename = "gbd.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
set_optimizer(gm, CPLEX_SILENT)
for bbl in gm.bbls
    if bbl isa BlackBoxRegressor
        bbl.equality=false
    end
end
set_param(gm, :ignore_feasibility, true)
set_param(gm, :ignore_accuracy, true)


# JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
# JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
# JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
# JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16

t1 = time()
find_bounds!(gm)
gm = test_survey_method(gm)
println("Elapsed time: ", time() - t1)
println("Optimal cost: $(gm.cost[end])")
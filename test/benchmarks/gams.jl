function test_solution_method(gm::GlobalModel = minlp(true))
    t1 = time()
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    update_vexity.(gm.bbls)  
    inadequately_feasible = findall(x -> x <= 0.15, feasibility(gm))
    if !isempty(inadequately_feasible)
        feasibility_sample.(gm.bbls[inadequately_feasible])
    end
    all(feasibility(gm) .>= 0.15) || 
        throw(OCTException("GlobalModel $(gm.name) still has inadequately feasible " *
                            "BBLs after sampling."))
    surveysolve(gm)
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while (gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end 
    @info "Optimization terminated."
    @info "Elapsed time: $(round(time()-t1, sigdigits=5)) seconds."
    @info "Optimal cost: $(round(gm.cost[end], sigdigits = 7))"
    return gm
end

filename = "weapons.gms"
filename = "himmel16.gms"
filename = "ramsey.gms"
filename = "alan.gms"
filename = "gbd.gms"
filename = "hydro.gms"
filename = "pollut.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
# gm = gear(true)
# gm = speed_reducer()
# gm = minlp(true)
set_param(gm, :ignore_feasibility, true)
set_param(gm, :ignore_accuracy, true)
set_optimizer(gm, CPLEX_SILENT)


if gm.name == "ramsey.gms"
    JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
    JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
elseif gm.name == "himmel16.gms"
    JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
    JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16
end

find_bounds!(gm)
test_solution_method(gm)
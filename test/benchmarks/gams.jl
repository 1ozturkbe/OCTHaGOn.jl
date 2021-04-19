function find_optimal_iteration(gm::GlobalModel)
    if any(bbl.equality == true for bbl in gm.bbls)
        @warn "GlobalModel $(gm.name) has equalities. These aren't guaranteed to be feasible. "
    end
    noteqs = [bbl.equality == false for bbl in gm.bbls]
    bbcs = [bbl isa BlackBoxClassifier for bbl in gm.bbls]
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if isempty(bbrs)
        for i = sortperm(gm.cost)
            feas = gm.feas_history[i] .* noteqs .* bbcs
            if !any(feas .<= -1e-10)
                @info "Iteration $(i) was optimal."
                return i
            end
        end
    else
        for i = sortperm(bbrs[1].actuals)
            feas = gm.feas_history[i] .* noteqs .* bbcs
            if !any(feas .<= -1e-10)
                @info "Iteration $(i) was optimal."
                return i
            end
        end
    end
    @info "None of the solutions fit the feasibility criteria. "
    return
end

function test_solution_method(gm::GlobalModel = minlp(true))
    t1 = time()
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    if !isempty(bbrs)
        update_vexity.(bbrs)  
    end
    inadequately_feasible = findall(x -> x <= 0.15, feasibility(gm))
    if !isempty(inadequately_feasible)
        feasibility_sample.(gm.bbls[inadequately_feasible])
    end
    all(feasibility(gm) .>= 0.15) || 
        throw(OCTException("GlobalModel $(gm.name) still has inadequately feasible " *
                            "BBLs after sampling."))
    surveysolve(gm)
    if !isempty(bbrs)
        update_vexity.(bbrs)
        update_leaf_vexity.(bbrs)  
    end
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while (gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end 
    if !isempty(bbrs)
        @assert length(bbrs) == 1
        bbr = bbrs[1]
        learn_constraint!(bbr, "upper" => minimum(bbr.actuals))
        update_tree_constraints!(gm, bbr)
        upper_bound_sample(bbr)
        learn_constraint!(bbr, "reg" => minimum(bbr.actuals), regression_sparsity = 0, max_depth = 5)
        update_tree_constraints!(gm, bbr)
        update_leaf_vexity(bbr)
        optimize!(gm)
        while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
            add_infeasibility_cuts!(gm)
            optimize!(gm)
        end
        while minimum(bbr.actuals) < bbr.thresholds[active_lower_tree(bbr)].second
            for bbc in bbcs # to clear already present cuts
                clear_tree_constraints!(gm, bbc)
                add_tree_constraints!(gm, bbc) 
            end
            learn_constraint!(bbr, "upper" => minimum(bbr.actuals))
            update_tree_constraints!(gm, bbr)
            upper_bound_sample(bbr)
            learn_constraint!(bbr, "reg" => minimum(bbr.actuals), regression_sparsity = 0, max_depth = 5)
            update_tree_constraints!(gm, bbr)
            update_leaf_vexity(bbr)
            try
                optimize!(gm)
            catch 
                break
            end
            while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
                add_infeasibility_cuts!(gm)
                try
                    optimize!(gm)
                catch 
                    break
                end
            end
        end 
        @info "Optimization terminated."
        it = find_optimal_iteration(gm)
        @info "Elapsed time: $(round(time()-t1, sigdigits=5)) seconds."
        @info "Optimal cost: $(round(bbr.actuals[it], sigdigits = 7))"
    else
        @info "Optimization terminated."
        it = find_optimal_iteration(gm)
        @info "Elapsed time: $(round(time()-t1, sigdigits=5)) seconds."
        @info "Optimal cost: $(round(gm.cost[it], sigdigits = 7))"
    end
    return
end

# filename = "weapons.gms"
# gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
# gm = gear(true)
# gm = speed_reducer()
gm = minlp(true)
set_param(gm, :ignore_feasibility, true)
set_param(gm, :ignore_accuracy, true)


if gm.name == "ramsey.gms"
    JuMP.set_upper_bound.(gm.vars, 4*ones(length(gm.vars))) # ramsey
    JuMP.set_lower_bound.(all_variables(gm.bbls), zeros(length(all_variables(gm.bbls)))) # ramsey
elseif gm.name == "himmel16.gms"
    JuMP.set_upper_bound.(gm.vars, ones(length(gm.vars))) # himmel16
    JuMP.set_lower_bound.(gm.vars, -ones(length(gm.vars))) # himmel16
end

find_bounds!(gm)
test_solution_method(gm)
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

# Iterative training algorithm
    # if !isempty(bbrs)
    #     @assert length(bbrs) == 1
    #     bbr = bbrs[1]
    #     learn_constraint!(bbr, "upper" => minimum(bbr.actuals))
    #     update_tree_constraints!(gm, bbr)
    #     upper_bound_sample(bbr)
    #     learn_constraint!(bbr, "reg" => minimum(bbr.actuals), regression_sparsity = 0, max_depth = 5)
    #     update_tree_constraints!(gm, bbr)
    #     update_leaf_vexity(bbr)
    #     optimize!(gm)
    #     while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
    #         add_infeasibility_cuts!(gm)
    #         optimize!(gm)
    #     end
    #     while minimum(bbr.actuals) < bbr.thresholds[active_lower_tree(bbr)].second
    #         for bbc in bbcs # to clear already present cuts
    #             clear_tree_constraints!(gm, bbc)
    #             add_tree_constraints!(gm, bbc) 
    #         end
    #         learn_constraint!(bbr, "upper" => minimum(bbr.actuals))
    #         update_tree_constraints!(gm, bbr)
    #         upper_bound_sample(bbr)
    #         learn_constraint!(bbr, "reg" => minimum(bbr.actuals), regression_sparsity = 0, max_depth = 5)
    #         update_tree_constraints!(gm, bbr)
    #         update_leaf_vexity(bbr)
    #         try
    #             optimize!(gm)
    #         catch 
    #             break
    #         end
    #         while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
    #             add_infeasibility_cuts!(gm)
    #             try
    #                 optimize!(gm)
    #             catch 
    #                 @info "Optimization terminated due to infeasibility."
    #                 break
    #             end
    #         end
    #     end 
    #     @info "Optimization terminated."
    #     it = find_optimal_iteration(gm)
    #     @info "Elapsed time: $(round(time()-t1, sigdigits=5)) seconds."
    #     @info "Optimal cost: $(round(bbr.actuals[it], sigdigits = 7))"
    # else
#=
refine:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-26
Refining of variable domains and solutions.
=#


"""
    find_bounds!(gm::GlobalModel; bbfs::Array{BlackBoxFunction} = [], M = 1e5, all_bounds::Bool=true)

Finds the outer variable bounds of GlobalModel by solving only over the linear constraints
and listed BBFs.
TODO: improve! Only find bounds of non-binary variables.
"""
function find_bounds!(gm::GlobalModel; bbfs::Array{BlackBoxFunction} = BlackBoxFunction[], M = 1e5, all_bounds::Bool=true)
    new_bounds = Dict(var => [-Inf, Inf] for var in gm.vars)
    # Finding bounds by min/maximizing each variable
    clear_tree_constraints!(gm)
    m = gm.model;
    x = gm.vars;
    old_bounds = get_bounds(m);
    orig_objective = JuMP.objective_function(gm.model)
    @showprogress 0.5 "Finding bounds..." for var in gm.vars
        if isinf(old_bounds[var][1]) || all_bounds
            @objective(m, Min, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][1] = getvalue(var);
            end
        end
        if isinf(old_bounds[var][2]) || all_bounds
            @objective(m, Max, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][2] = getvalue(var);
            end
        end
    end
    # Revert objective
    @objective(m, Min, orig_objective)
    # Warning in case Inf bounds remain
    bound!(gm, new_bounds)
    unbounds = get_unbounds(gm)
#     if !isempty(unbounds)
#         @warn("Unbounded variables in GlobalModel " * gm.name * " in BlackBoxFunctions...")
#         @warn("Will try to tighten bounds through an exponential search, with M = " * string(M) * ".")
#         unbounds = get_unbounds(gm)
#         bbf_to_var = match_bbfs_to_vars(gm.bbfs, flat(keys(unbounds)))
#         for (bbf, unbounded_vars) in bbf_to_var
#             last_unbounds = Dict(var => local_bounds[var] for var in unbounded_vars)
#             for (var, bounds) in last_unbounds
#                 JuMP.set_lower_bound(var, 0)
#                 JuMP.set_upper_bound(var, 1)
#             end
#             df = boundary_sample(bbf, fraction = 0.5)
#             append!(df, lh_sample(bbf, iterations = 1, n_samples = bbf.n_samples - size(df, 1)), cols=:setequal)
#             for (var, bounds) in last_unbounds # Revert bounds
#                 JuMP.set_lower_bound(var, minimum(bounds))
#                 JuMP.set_upper_bound(var, maximum(bounds))
#                 if all(isinf.(bounds)) # Log-scale the samples
#                     df[!,string(var)] = 2. .* (rand(bbf.n_samples) .- 0.5) .* M.^ df[!,string(var)]
#                 elseif isinf(minimum(bounds))
#                     df[!,string(var)] = maximum(bounds) - M.^ df[!,string(var)]
#                 elseif isinf(maximum(bounds))
#                     df[!,string(var)] = minimum(bounds) - M.^ df[!,string(var)]
#                 end
#             end
#             # Find tightest enclosing box
#             eval!(bbf, df)
#             df = knn_sample(bbf)
#             eval!(bbf, df)
#             df = knn_sample(bbf)
#             eval!(bbf, df)
#             feas_X = bbf.X[findall(x -> x .>= 0, bbf.Y),:];
#             box_bounds = Dict(bbf.vars[i] => [minimum(feas_X[!,i]), maximum(feas_X[!,i])] for i=1:length(bbf.vars))
#         end
#     end
    return
end
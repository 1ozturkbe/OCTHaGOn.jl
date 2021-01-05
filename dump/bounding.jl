gm = minlp(true)
bounds = get_bounds(gm)
lin_bounds = find_linear_bounds!(gm)
unboundeds = get_unbounds(gm)
unbounded_vars = collect(keys(unboundeds))
matching = match_bbfs_to_vars(gm, unbounded_vars);

M=1e3
bbf = matching[1].first
# function exponential_bound(bbf::BlackBoxFunction, M = max_iterations::Int64 = 3)
    unbounds = get_unbounds(bbf)
    for (var, bounds) in unbounds
        JuMP.set_lower_bound(var, 0)
        JuMP.set_upper_bound(var, 1)
    end
    df = boundary_sample(bbf, fraction = 0.5)
    append!(df, lh_sample(bbf, iterations = 1, n_samples = get_param(bbf, :n_samples) - size(df, 1)), cols=:setequal)
    for (var, bounds) in unbounds # Revert bounds
        !isinf(minimum(bounds)) && JuMP.set_lower_bound(var, minimum(bounds))
        !isinf(maximum(bounds)) && JuMP.set_upper_bound(var, maximum(bounds))
        if all(isinf.(bounds)) # Log-scale the samples
            df[!,string(var)] = 2. .* (rand(get_param(bbf, :n_samples)) .- 0.5) .* M.^ df[!,string(var)]
        elseif isinf(minimum(bounds))
            df[!,string(var)] = maximum(bounds) - M.^ df[!,string(var)]
        elseif isinf(maximum(bounds))
            df[!,string(var)] = minimum(bounds) - M.^ df[!,string(var)]
        end
    end
    eval!(bbf, df)
    df = knn_sample(bbf)
    eval!(bbf, df)
    df = knn_sample(bbf)
    eval!(bbf, df)
    feas_X = bbf.X[findall(x -> x .>= 0, bbf.Y),:];
            # Find tightest enclosing box
            box_bounds = Dict(bbf.vars[i] => [minimum(feas_X[!,i]), maximum(feas_X[!,i])] for i=1:length(bbf.vars))
#         end
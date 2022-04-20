"""
    truncate_sigfigs(data, digits = 8)

Cuts off more than digits significant figures. 
Used to reduce tree training degeneracies. 
"""
function truncate_sigfigs(data, digits = 7)
    return round.(data, sigdigits = digits)
end

""" Computes the bounds of the union of variables in a BlackBoxLearner and its LinkedLearners. """
function get_joint_bounds(bbl::BlackBoxLearner)
    !isempty(bbl.lls) || throw(OCTHaGOnException("BlackBoxLearner $(bbl.name) must have non-zero LinkedLearners for computing joint bounds."))
    bounds = Dict(var => [] for var in bbl.vars)
    bbl_bounds = flattened_bounds(bbl)
    ll_bounds = [flattened_bounds(ll) for ll in bbl.lls]
    for i = 1:length(bbl.vars)
        var = bbl.vars[i]
        var_bound = [
            minimum([minimum(bbl_bounds[i]), 
                    [minimum(ll_bound[i]) for ll_bound in ll_bounds]...])
            maximum([maximum(bbl_bounds[i]), 
                    [maximum(ll_bound[i]) for ll_bound in ll_bounds]...])]
        bounds[var] = var_bound
    end
    return bounds
end

"""
    $(TYPEDSIGNATURES)

Uniformly Latin Hypercube samples the variables of GlobalModel, as long as all
lbs and ubs are defined.
"""
function lh_sample(bounds::Dict; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    check_bounds(bounds)
    vars = flat(keys(bounds))
    n_dims = length(vars)
    if lh_iterations > 0
        plan, _ = LHCoptim(n_samples, n_dims, lh_iterations);
    else
        plan = rand(n_samples, n_dims)
    end
   X = scaleLHC(plan,[(minimum(bounds[var]), maximum(bounds[var])) for var in vars]);
   return DataFrame(truncate_sigfigs(X), string.(vars))
end

function lh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    if isempty(bbl.lls)
        bounds = get_bounds(bbl.vars)
        return lh_sample(bounds; lh_iterations = lh_iterations, n_samples = n_samples)
    else
        bounds = get_joint_bounds(bbl)
        return lh_sample(bounds; lh_iterations = lh_iterations, n_samples = n_samples)
    end
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

"""
    $(TYPEDSIGNATURES)

Samples a BlackBoxLearner on the corners of the variable hypercube. 
Samples a subset of corners for learners with large number of variables. 
"""
function boundary_sample(bounds::Dict; n_samples::Int64 = 100, fraction::Float64 = 0.5,
                         warn_string::String = "")
    check_bounds(bounds);
    vars = flat(keys(bounds))
    n_vars = length(vars);
    vks = string.(vars);
    lbs = DataFrame(Dict(string(key) => minimum(val) for (key, val) in bounds))
    ubs = DataFrame(Dict(string(key) => maximum(val) for (key, val) in bounds))
    n_comb = 0
    if n_vars <= 12
        n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    else
        n_comb = 1e6
    end
    nX = DataFrame(vks .=> [Float64[] for i in vks])
    sample_indices = [];
    if n_comb >= fraction*n_samples 
        n_comb = 2*n_vars+2; # Everything is double because we choose min's and max's
        choosing = 1;
        while n_comb <= fraction*n_samples
            choosing = choosing + 1;
            n_comb += 2*choose(n_vars, choosing);
        end
        choosing = choosing - 1; # Determined maximum 'choose' coefficient
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:choosing); # Choose 1 and above
    else
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:n_vars); # Choose 1 and above
    end
    for i in sample_indices
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        lbscopy[:, vks[i]] = ubscopy[:, vks[i]];
        append!(nX, lbscopy);
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        ubscopy[:, vks[i]] = lbscopy[:, vks[i]];
        append!(nX, ubscopy);
    end
    return nX # Note: boundary samples are not truncated for significant figures
end

function boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.1)
    if isempty(bbl.lls)
        return boundary_sample(get_bounds(bbl.vars), n_samples = get_param(bbl, :n_samples), fraction = fraction,
                           warn_string = bbl.name)
    else
        bounds = get_joint_bounds(bbl)
        return boundary_sample(bounds, n_samples = get_param(bbl, :n_samples), fraction = fraction,
                            warn_string = bbl.name)
    end
end


"""
    knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10, sample_density = 1e-5, sample_idxs = nothing)

Implements KNN sampling on a BlackBoxLearner. 
Note: must have at least one in/feasible point for the algorithm. 
"""
function knn_sample(bbl::BlackBoxClassifier; k::Int64 = 10, sample_density = 1e-5, sample_idxs = nothing, sign = 1)
    if bbl.feas_ratio == 0. || bbl.feas_ratio == 1.0
        throw(OCTHaGOnException("Constraint " * string(bbl.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    vks = string.(bbl.vars)
    df = DataFrame(vks .=> [Float64[] for i in vks])
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    feas_class = classify_patches(bbl, idxs);
    negatives = findall(x -> x .< 0, bbl.Y)
    if !isnothing(sample_idxs)
        negatives = intersect(negatives, sample_idxs)
    end
    for i = 1:length(negatives) # This loop is for making sure that every possible root is sampled only once.
        if feas_class[negatives[i]] == "mixed"
            nodes = [idxs[negatives[i]][j] for j=1:length(idxs[negatives[i]]) 
                            if (bbl.Y[idxs[negatives[i]][j]] >= 0 && dists[negatives[i]][j] >= sample_density)]
            push!(nodes, negatives[i])
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np)
        end
    end
    return truncate_sigfigs(df)
end

"""
    $(TYPEDSIGNATURES)

Samples and evaluates nonlinear constraints using full suite of methods. 
"""
function uniform_sample_and_eval!(bbl::BlackBoxLearner;
                          boundary_fraction::Float64 = 0.1,
                          lh_iterations::Int64 = 0, sample_density = 1.e-5)
    @assert size(bbl.X, 1) == 0 # TODO: fix this w.r.t. data-driven constraints. 
    vks = string.(bbl.vars)
    n_dims = length(vks);
    check_bounds(get_bounds(bbl))
    df = boundary_sample(bbl, fraction = boundary_fraction)
    eval!(bbl, df)
    df = lh_sample(bbl, lh_iterations = lh_iterations, n_samples = get_param(bbl, :n_samples) - size(df, 1))
    eval!(bbl, df);
    if bbl isa BlackBoxClassifier
        if bbl.feas_ratio == 1.0
            @info(string(bbl.name) * " was not KNN sampled since it has no infeasible samples.")
        elseif bbl.feas_ratio == 0.0
            throw(OCTHaGOnException(string(bbl.name) * " has zero feasible samples. " *
                               "Please find at least one feasible sample, seed the data and KNN sample."))
        else
            df = knn_sample(bbl, k= maximum([10, 2*length(bbl.vars) + 1]), sample_density = sample_density)
            if size(df, 1) > 0
                eval!(bbl, df)
            end
        end
    end
    bbl.max_Y = maximum(filter(!isinf, bbl.Y))
    bbl.min_Y = minimum(filter(!isinf, bbl.Y))
    # TODO: update these mx/min Y more frequently. 
    # Setting vague bounds for the dependent variable. 
    # This is the only big-M required in the formulation, adding 50% margin to all observed Y values. 
    if bbl isa BlackBoxRegressor
        max_Y = bbl.max_Y
        min_Y = bbl.min_Y
        lower_margined_bound = min_Y - (max_Y - min_Y)/2
        upper_margined_bound = max_Y + (max_Y - min_Y)/2
        if !JuMP.has_lower_bound(bbl.dependent_var)
            JuMP.set_lower_bound(bbl.dependent_var, lower_margined_bound)
        end
        if !JuMP.has_upper_bound(bbl.dependent_var)
            JuMP.set_upper_bound(bbl.dependent_var, upper_margined_bound)
        end
        for ll in bbl.lls
            if !JuMP.has_lower_bound(ll.dependent_var)
                JuMP.set_lower_bound(ll.dependent_var, lower_margined_bound)
            end
            if !JuMP.has_upper_bound(ll.dependent_var)
                JuMP.set_upper_bound(ll.dependent_var, upper_margined_bound)
            end
        end
    end
    return 
end

function uniform_sample_and_eval!(bbls::Array{BlackBoxLearner}; lh_iterations = 0, sample_density = 1e-5) 
    for bbl in bbls 
        uniform_sample_and_eval!(bbl, lh_iterations = lh_iterations, sample_density = sample_density)
    end
    return
end

function uniform_sample_and_eval!(gm::GlobalModel)
    @info "Sampling..."
    uniform_sample_and_eval!(gm.bbls; lh_iterations = get_param(gm, :lh_iterations), sample_density = get_param(gm, :sample_density))
    return
end

"""
    $(TYPEDSIGNATURES)

Gets Latin Hypercube samples that fall in the leaf of the last solution.
"""
function last_leaf_sample(bbc::BlackBoxClassifier, n_samples = get_param(bbc, :n_samples))
    isempty(bbc.active_leaves) &&         
        throw(OCTHaGOnException("BBC $(bbc.name) needs to be optimized first, to figure out its active leaves."))
    if !bbc.equality
        last_leaf = bbc.active_leaves[1]
        idxs = findall(x -> x .>= 0.5, IAI.apply(bbc.learners[end], bbc.X) .== last_leaf)
        lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbc.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbc.vars))
    else
        for last_leaf in bbc.active_leaves
            idxs = findall(x -> x .>= 0.5, IAI.apply(bbc.learners[end], bbc.X) .== last_leaf)
            lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
            ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
            plan, _ = LHCoptim(Int(ceil(n_samples/2)), length(bbc.vars), 3);
            X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
            return DataFrame(truncate_sigfigs(X), string.(bbc.vars))
        end
    end
end

function last_leaf_sample(bbr::BlackBoxRegressor, n_samples = get_param(bbr, :n_samples))
    if isempty(bbr.active_leaves)
        throw(OCTHaGOnException("BBR $(bbr.name) needs to be optimized first, to figure out its active leaves."))
    elseif length(bbr.active_trees) == 2
        upper_leaf, lower_leaf = sort(bbr.active_leaves)
        upper_leafneighbor = [];
        lower_leafneighbor = [];
        for (tree_idx, threshold) in bbr.active_trees
            if threshold.first == "upper"
                append!(upper_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== -upper_leaf)
            elseif threshold.first == "lower"
                append!(lower_leafneighbor, IAI.apply(bbr.learners[tree_idx], bbr.X) .== lower_leaf)
            end
        end
        idxs =  findall(x -> x .>= 0.5, upper_leafneighbor .* lower_leafneighbor)
        if length(idxs) == 0
            @info("No points in $(bbr.name) in the intersection of trees. Widening sampling. ")
            idxs =  findall(x -> x .>= 0.5, upper_leafneighbor + lower_leafneighbor)
        end
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbr.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbr.vars))
    elseif length(bbr.active_trees) == 1
        leaf = bbr.active_leaves[1]
        tree = bbr.learners[collect(keys(bbr.active_trees))[1]]
        idxs = findall(x -> x .== leaf, IAI.apply(tree, bbr.X))
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(n_samples, length(bbr.vars), 3);
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        return DataFrame(truncate_sigfigs(X), string.(bbr.vars))
    else 
        throw(OCTHaGOnException("No active trees found in BBR $(bbr.name) while attempting to leaf-sample."))
    end
end

"""
    feasibility_sample(gm::GlobalModel)
    feasibility_sample(bbc::BlackBoxClassifier, n_samples::Int64 = get_param(bbc, :n_samples))

Equally LH samples the leaves of BBC with not enough feasible samples. 
"""
function feasibility_sample(bbc::BlackBoxClassifier, n_samples::Int64 = get_param(bbc, :n_samples))
    minbucket = minimum([Int64(round(bbc.feas_ratio .* size(bbc.X, 1))), Int64(floor(0.05*size(bbc.X, 1)))])
    lnr = base_classifier()
    IAI.set_params!(lnr, minbucket = minbucket)
    lnr = learn_from_data!(bbc.X, bbc.Y .>= 0, lnr; fit_classifier_kwargs()...)
    all_leaves = find_leaves(lnr)
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    orig_feasratio = copy(bbc.feas_ratio)
    leaf_idxs = IAI.apply(lnr, bbc.X)
    for leaf in feas_leaves
        idxs = findall(x -> x .>= 0.5, leaf_idxs .== leaf)
        lbs = [minimum(col) for col in eachcol(bbc.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbc.X[idxs, :])]
        plan, _ = LHCoptim(Int64(ceil(n_samples / length(feas_leaves))), length(bbc.vars), 3);  
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        eval!(bbc, DataFrame(X, string.(bbc.vars)))
    end    
    if bbc.feas_ratio >= get_param(bbc, :threshold_feasibility)
        @info("BBC $(bbc.name) has passed threshold feasibility $(round(get_param(bbc, :threshold_feasibility), sigdigits=3)), " * 
              "from $(round(orig_feasratio, sigdigits=3)) to $(round(bbc.feas_ratio, sigdigits=3)).")
    elseif bbc.feas_ratio > orig_feasratio
        @info("BBC $(bbc.name) has improved feasibility from $(round(orig_feasratio, sigdigits=3)) to $(round(bbc.feas_ratio, sigdigits=3)), " * 
                "but still not over threshold $(round(get_param(bbc, :threshold_feasibility), sigdigits=3)).")
    else
        @info("BBC $(bbc.name) feasibility improvement failed! Please check your functions.")
    end
    return
end

function feasibility_sample(gm::GlobalModel)
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier && bbl.feas_ratio <= get_param(bbl, :threshold_feasibility)
            feasibility_sample(bbl)
        end
    end
    return
end

""" 
    upper_bound_sample(bbr::BlackBoxRegressor)

Samples within the feasible regions of the active upper bounding tree.
"""
function upper_bound_sample(bbr::BlackBoxRegressor)
    @assert !isempty(bbr.active_trees) # We must have optimized at least once over the old tree. 
    active_upper_idx = active_upper_tree(bbr)
    lnr = bbr.learners[active_upper_idx]
    @assert lnr isa IAI.OptimalTreeClassifier
    all_leaves = find_leaves(lnr)
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
    leaf_idxs = IAI.apply(lnr, bbr.X)
    n_samples = get_param(bbr, :n_samples)
    for leaf in feas_leaves
        idxs = findall(x -> x == leaf, leaf_idxs)
        lbs = [minimum(col) for col in eachcol(bbr.X[idxs, :])]
        ubs = [maximum(col) for col in eachcol(bbr.X[idxs, :])]
        plan, _ = LHCoptim(Int64(ceil(n_samples / length(feas_leaves))), length(bbr.vars), 3);  
        X = scaleLHC(plan, [(lbs[i], ubs[i]) for i=1:length(lbs)]);
        eval!(bbr, DataFrame(X, string.(bbr.vars)))
    end
    return
end


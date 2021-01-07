"""
    lh_sample(vars::Array{JuMP.VariableRef, 1}; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    lh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)

Uniformly Latin Hypercube samples the variables of GlobalModel, as long as all
lbs and ubs are defined.
"""
function lh_sample(vars::Array{JuMP.VariableRef, 1}; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
    bounds = get_bounds(vars)
    check_bounds(bounds)
    n_dims = length(vars)
    if lh_iterations > 0
        plan, _ = LHCoptim(n_samples, n_dims, lh_iterations);
    else
        plan = rand(n_samples, n_dims)
    end
   X = scaleLHC(plan,[(minimum(bounds[var]), maximum(bounds[var])) for var in vars]);
   return DataFrame(X, string.(vars))
end

function lh_sample(bbl::BlackBoxLearner; lh_iterations::Int64 = 0,
                   n_samples::Int64 = 1000)
   return lh_sample(bbl.vars; lh_iterations = lh_iterations, n_samples = n_samples)
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

"""
    boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.5)
    boundary_sample(vars::Array{JuMP.VariableRef, 1}; n_samples = 100, fraction::Float64 = 0.5,
                         warn_string::String = "")

*Smartly* samples the constraint along the variable boundaries.
    NOTE: Because we are sampling symmetrically for lower and upper bounds,
    the choose coefficient has to be less than ceil(half of number of dims).
"""
function boundary_sample(vars::Array{JuMP.VariableRef, 1}; n_samples::Int64 = 100, fraction::Float64 = 0.5,
                         warn_string::String = "")
    bounds = get_bounds(vars);
    check_bounds(bounds);
    n_vars = length(vars);
    vks = string.(vars);
    lbs = DataFrame(Dict(string(key) => minimum(val) for (key, val) in bounds))
    ubs = DataFrame(Dict(string(key) => maximum(val) for (key, val) in bounds))
    n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    nX = DataFrame([Float64 for i in vks], vks)
    sample_indices = [];
    if n_comb >= fraction*n_samples
        @warn("Can't exhaustively sample the boundary of Constraint " * string(warn_string) * ".")
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
    return nX
end

function boundary_sample(bbl::BlackBoxLearner; fraction::Float64 = 0.5)
    return boundary_sample(bbl.vars, n_samples = get_param(bbl, :n_samples), fraction = fraction,
                           warn_string = bbl.name)
end


"""
Does KNN and interval arithmetic based sampling once there is at least one feasible
    sample to a BlackBoxLearner.
"""
function knn_sample(bbl::BlackBoxLearner; k::Int64 = 10)
    if bbl.feas_ratio == 0. || bbl.feas_ratio == 1.0
        throw(OCTException("Constraint " * string(bbl.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    vks = string.(bbl.vars)
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    positives = findall(x -> x .>= 0 , bbl.Y);
    feas_class = classify_patches(bbl, idxs);
    for center_node in positives # This loop is for making sure that every possible root is sampled only once.
        if feas_class[center_node] == "mixed"
            nodes = [idx for idx in idxs[center_node] if bbl.Y[idx] < 0];
            push!(nodes, center_node)
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np);
        end
    end
    return df
end

"""
    uniform_sample_and_eval!(bbl::Union{BlackBoxLearner, GlobalModel, Array{BlackBoxLearner}};
                              boundary_fraction::Float64 = 0.5,
                              lh_iterations::Int64 = 0)

Uniform samples and evaluates a BlackBoxLearner.
Keyword arguments:
    boundary_fraction: maximum ratio of boundary samples
    lh_iterations: number of GA populations for LHC sampling (0 is a random LH.)
"""
function uniform_sample_and_eval!(bbl::BlackBoxLearner;
                          boundary_fraction::Float64 = 0.5,
                          lh_iterations::Int64 = 0)
    @assert size(bbl.X, 1) == 0 
    vks = string.(bbl.vars)
    n_dims = length(vks);
    check_bounds(get_bounds(bbl))
    df = boundary_sample(bbl, fraction = boundary_fraction)
    eval!(bbl, df)
    df = lh_sample(bbl, lh_iterations = lh_iterations, n_samples = get_param(bbl, :n_samples) - size(df, 1))
    eval!(bbl, df);
    if bbl isa BlackBoxClassifier
        if bbl.feas_ratio == 1.0
            @warn(string(bbl.name) * " was not KNN sampled since it has no infeasible samples.")
        elseif bbl.feas_ratio == 0.0
            throw(OCTException(string(bbl.name) * " has zero infeasible samples. " *
                               "Please find at least one feasible sample, seed the data and KNN sample."))
        else
            df = knn_sample(bbl)
            eval!(bbl, df)
        end
    end
    return
end

function uniform_sample_and_eval!(bbls::Array{BlackBoxLearner}; lh_iterations = 0) 
    for bbl in bbls 
        uniform_sample_and_eval!(bbl, lh_iterations = lh_iterations)
    end
    return
end

uniform_sample_and_eval!(gm::GlobalModel) = uniform_sample_and_eval!(gm.bbls; lh_iterations = get_param(gm, :lh_iterations))
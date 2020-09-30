function set_objective!(model, c, vars, vks)
    @objective(model, Min, sum(c .* [vars[vk] for vk in vks]));
    return
end

function InitModel(c, vks, lbs, ubs)
    m = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0))
    vars = @variable(m, x[vks])
    bound!(JuMP.all_variables(m), vks, lbs, ubs)
    set_objective!(m, c, vars, vks)
    return m
end

@with_kw mutable struct ModelData
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::Union{Symbol, String} = "Model"                               # Example name
    c::Array                                                            # Cost vector
    fns::Array{BlackBoxFunction} = Array{BlackBoxFunction}[]            # Black box (>/= 0) functions
    lin_constrs::Array{Union{ScalarConstraint, VectorConstraint}} = []  # Linear constraints
    vks::Dict{Symbol, Tuple} = Dict(:x => (length(c)))                  # Varkeys and size
    lbs::Dict{Symbol, Any} = Dict(vk .=> -Inf.*ones(size(tup)) for (vk, tup) in vks)  # Variable lower bounds
    ubs::Dict{Symbol, Any} = Dict(vk .=> Inf.*ones(size(tup)) for (vk, tup) in vks)  # Variable upper bounds
    model::JuMP.Model = InitModel(c, vks, lbs, ubs)                     # JuMP model
    vars = model[:x]                                                    # JuMP variables
end

function (md::ModelData)(name::Union{String, Int64})
    """ Calls a BlackBoxFunction in ModelData by name. """
    fn_names = getfield.(md.fns, :name);
    fns = md.fns[findall(x -> x == name, fn_names)];
    if length(fns) == 1
        return fns[1]
    elseif length(fns) == 0
        @warn("No constraint with name ", name)
        return
    else
        @warn("Multiple constraints with name ", name)
        return fns
    end
end

function feasibility(bbf::Union{ModelData, Array{BlackBoxFunction}, BlackBoxFunction})
    """ Returns the feasibility of data points in a BBF or MD. """
    if isa(bbf, BlackBoxFunction)
        return bbf.feas_ratio
    elseif isa(bbf, Array{BlackBoxFunction})
        return [feasibility(fn) for fn in bbf]
    else
        return [feasibility(fn) for fn in bbf.fns]
    end
end

function accuracy(bbf::Union{ModelData, BlackBoxFunction})
    """ Returns the accuracy of learners in a BBF or MD. """
    if isa(bbf, BlackBoxFunction)
        if bbf.feas_ratio in [1., 0]
            @warn(string("Accuracy of BlackBoxFunction ", bbf.name, " is tautological."))
            return 1.
        elseif isempty(bbf.learners)
            throw(OCTException(string("BlackBoxFunction ", bbf.name, " has not been trained yet.")))
        else
            return bbf.accuracies[end]
        end
    else
        return [accuracy(fn) for fn in bbf.fns]
    end
end

function add_fn!(md::ModelData, bbf::BlackBoxFunction)
    update_bounds!(bbf, lbs = md.lbs, ubs = md.ubs)
    update_bounds!(md, lbs = bbf.lbs, ubs = bbf.ubs) # TODO: optimize
    if bbf.name == ""
        bbf.name = length(md.fns)+1;
    end
    push!(md.fns, bbf)
end

function add_lin_constr!(md::ModelData, constraint)
    """ Adds a linear JuMP constraint to md.
    Note: proper syntax is 'add_lin_constr!(md, @build_constraint(...))'
    """
    if isa(constraint, Array)
        append!(md.lin_constrs, constraint)
        add_constraint.(md.model, constraint)
    else
        push!(md.lin_constrs, constraint)
        add_constraint(md.model, constraint)
    end

end

function get_max(a, b)
    return maximum([a,b])
end

function get_min(a,b)
    return minimum([a,b])
end

function update_bounds!(md::Union{ModelData, BlackBoxFunction, Array{BlackBoxFunction}};
                        lbs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}(),
                        ubs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}())
    """ Updates the outer bounds of ModelData and its BlackBoxFunctions, or the bounds
    of a single BlackBoxFunction. """
    if isa(md, Array{BlackBoxFunction})
        for fn in md
            update_bounds!(fn, lbs=lbs, ubs=ubs)
        end
        return
    end
    nlbs = merge(get_max, md.lbs, lbs);
    nubs = merge(get_min, md.ubs, ubs)
    if any([nlbs[vk] .> nubs[vk] for vk in md.vks])
        throw(OCTException("Infeasible bounds."))
    end
    md.lbs = Dict(vk => nlbs[vk] for vk in keys(md.lbs));
    md.ubs = Dict(vk => nubs[vk] for vk in keys(md.ubs));
    if isa(md, ModelData)
        for fn in md.fns
            update_bounds!(fn, lbs = Dict(vk => md.lbs[vk] for vk in fn.vks),
                               ubs = Dict(vk => md.ubs[vk] for vk in fn.vks)); # TODO: check that it works.
        end
        bound!(md.vars, md.vks, md.lbs, md.ubs)
    end
    return
end

function check_bounds(md::Union{ModelData, BlackBoxFunction})
    """ Checks outer-boundedness of BBF or MD. """
    if isa(md, BlackBoxFunction)
        if any(isinf.(values(md.lbs))) || any(isinf.(values(md.ubs)))
            return false
        else
            return true
        end
    else
        return any(check_bounds.(md.fns))
    end
end

function lh_sample(md::Union{ModelData,BlackBoxFunction}; iterations::Int64 = 3,
                n_samples::Int64 = 1000)
"""
Uniformly Latin Hypercube samples the variables of ModelData, as long as all
lbs and ubs are defined.
"""
   if any(isinf.(values(md.lbs))) || any(isinf.(values(md.ubs)))
       throw(OCTException("Model is not properly bounded."))
   end
   n_dims = length(md.vks)
   plan, _ = LHCoptim(n_samples, n_dims, iterations);
   X = scaleLHC(plan,[(md.lbs[vk], md.ubs[vk]) for vk in md.vks]);
   return DataFrame(X, md.vks)
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

function boundary_sample(bbf::Union{ModelData,BlackBoxFunction}; fraction::Float64 = 0.5)
""" *Smartly* samples the constraint along the variable boundaries.
    NOTE: Because we are sampling symmetrically for lower and upper bounds,
    the choose coefficient has to be less than ceil(half of number of dims). """
    n_vars = length(bbf.vks)
    n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    nX = DataFrame([Float64 for i in bbf.vks], bbf.vks)
    if isa(bbf, BlackBoxFunction) && n_comb >= fraction*bbf.n_samples
        @warn("Can't exhaustively sample the boundary of Constraint " * string(bbf.name) * ".")
        n_comb = 2*n_vars+2; # Everything is double because we choose min's and max's
        choosing = 1;
        while n_comb <= fraction*bbf.n_samples
            choosing = choosing + 1;
            n_comb += 2*choose(n_vars, choosing);
        end
        choosing = choosing - 1; # Determined maximum 'choose' coefficient
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:choosing); # Choose 1 and above
        for i in sample_indices
            lbs = DataFrame(bbf.lbs); ubs = DataFrame(bbf.ubs);
            lbs[:, bbf.vks[i]] = ubs[:, bbf.vks[i]];
            append!(nX, lbs);
            lbs = DataFrame(bbf.lbs); ubs = DataFrame(bbf.ubs);
            ubs[:, bbf.vks[i]] = lbs[:, bbf.vks[i]];
            append!(nX, ubs);
        end
        return nX
    else
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:n_vars); # Choose 1 and above
        for i in sample_indices
            lbs = DataFrame(bbf.lbs); ubs = DataFrame(bbf.ubs);
            lbs[:, bbf.vks[i]] = ubs[:, bbf.vks[i]];
            append!(nX, lbs);
        end
        return nX
    end
end

function knn_sample(bbf::BlackBoxFunction; k::Int64 = 15)
    """ Does KNN and interval arithmetic based sampling once there is at least one feasible
        sample to a BlackBoxFunction. """
    if bbf.feas_ratio == 0. || bbf.feas_ratio == 1.0
        throw(OCTException("Constraint " * string(bbf.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    df = DataFrame([Float64 for i in bbf.vks], bbf.vks)
    build_knn_tree(bbf);
    idxs, dists = find_knn(bbf, k=k);
    positives = findall(x -> x .>= 0 , bbf.Y);
    feas_class = classify_patches(bbf, idxs);
    for center_node in positives # This loop is for making sure that every possible root is sampled only once.
        if feas_class[center_node] == "mixed"
            nodes = [idx for idx in idxs[center_node] if bbf.Y[idx] < 0];
            push!(nodes, center_node)
            np = secant_method(bbf.X[nodes, :], bbf.Y[nodes, :])
            append!(df, np);
        end
    end
    return df
end

function sample_and_eval!(bbf::Union{BlackBoxFunction, Array{BlackBoxFunction}, ModelData};
                          n_samples:: Union{Int64, Nothing} = nothing,
                          boundary_fraction::Float64 = 0.5,
                          iterations::Int64 = 3)
    """ Samples and evaluates BlackBoxFunction, with n_samples new samples.
    Arguments
    n_samples: number of samples, overwrites bbf.n_samples.
    boundary_fraction: maximum ratio of boundary samples
    iterations: number of GA populations for LHC sampling (0 is a random LH.)
    ratio:
    If there is an optimized gp, ratio*n_samples is how many random LHC samples are generated
    for prediction from GP. """
    if isa(bbf, ModelData)
        for fn in bbf.fns
            sample_and_eval!(fn, n_samples = n_samples, boundary_fraction = boundary_fraction,
                             iterations = iterations);
        end
        return
    elseif isa(bbf, Array{BlackBoxFunction})
        for fn in bbf
            sample_and_eval!(fn, n_samples = n_samples, boundary_fraction = boundary_fraction,
                             iterations = iterations);
        end
        return
    end
    if !isnothing(n_samples)
        bbf.n_samples = n_samples
    end
    vks = bbf.vks;
    n_dims = length(vks);
    if size(bbf.X,1) == 0 # If we don't have data yet, uniform and boundary sample.
       df = boundary_sample(bbf, fraction = boundary_fraction)
       eval!(bbf, df)
       df = lh_sample(bbf, iterations = iterations, n_samples = bbf.n_samples - size(df, 1))
       eval!(bbf, df);
    elseif bbf.feas_ratio == 1.0
        @warn(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.")
    elseif bbf.feas_ratio == 0.0
        throw(OCTException(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.
                            Please find at least one feasible sample and try again. "))
    else                  # otherwise, KNN sample!
        df = knn_sample(bbf)
        eval!(bbf, df)
    end
    return
end

function globalsolve(md::ModelData)
    """ Creates and solves the global optimization model using the linear constraints from ModelData,
        and approximated nonlinear constraints from inside its BlackBoxFunctions."""
    clear_tree_constraints!(md); # remove trees from previous solve (if any).
    add_tree_constraints!(md); # refresh latest tree constraints.
    status = JuMP.optimize!(md.model);
    return status
end

function solution(md::ModelData)
    """ Returns the optimal solution of the solved global optimization model. """
    vals = getvalue.(md.vars)
    return DataFrame(Dict(vk => vals[vk] for vk in md.vks))
end

function evaluate_feasibility(md::ModelData)
    """ Evaluates each constraint at solution to make sure it is feasible. """
    soln = solution(md);
    feas = [];
    for fn in md.fns
        eval!(fn, soln)
    end
    fn_names = getfield.(md.fns, :name);
    infeas_idxs = fn_names[findall(vk -> md(vk).Y[end] .< 0, fn_names)]
    feas_idxs = fn_names[findall(vk -> md(vk).Y[end] .>= 0, fn_names)]
    return feas_idxs, infeas_idxs
end

function find_bounds!(md::ModelData; all_bounds=true)
    """Finds the outer variable bounds of ModelData by solving over the linear constraints. """
    ubs = Dict(md.vks .=> Inf)
    lbs = Dict(md.vks .=> -Inf)
    # Finding bounds by min/maximizing each variable
    m = md.model;
    x = md.vars;
    @showprogress 0.5 "Finding bounds..." for vk in md.vks
        if isinf(md.lbs[vk]) || all_bounds
            @objective(m, Min, x[vk]);
            JuMP.optimize!(m);
            lbs[vk] = getvalue.(x)[vk];
        end
        if isinf(md.lbs[vk]) || all_bounds
            @objective(m, Max, x[vk]);
            JuMP.optimize!(m);
            ubs[vk] = getvalue.(x)[vk];
        end
    end
    set_objective!(md.model, md.c, md.vars, md.vks) # revert objective
    update_bounds!(md, lbs=lbs, ubs=ubs)
    return
end

function import_trees(dir, md::ModelData)
    """ Returns trees trained over given ModelData,
    where filename points to the model name. """
    trees = [IAI.read_json(string(dir, "_tree_", i, ".json")) for i=1:length(md.fns)];
    return trees
end
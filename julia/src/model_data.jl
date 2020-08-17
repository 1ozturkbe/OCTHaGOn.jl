using Combinatorics
using DataFrames
using Gurobi
using JuMP
using MathOptInterface
using MosekTools
using NearestNeighbors
using LatinHypercubeSampling
using Parameters
using SparseArrays
using Random
Random.seed!(1);

include("exceptions.jl");
include("black_box_function.jl");
include("root_finding.jl");

@with_kw mutable struct ModelData
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::String = "Model"                                 # Example name
    c::Array                                               # Cost vector
    fns::Array{BlackBoxFunction} = Array{BlackBoxFunction}[]           # Black box (>/= 0) functions
    ineqs_A::Array = SparseMatrixCSC[]                     # Linear inequality A vector, in b-Ax>=0
    ineqs_b::Array = Array[]                               # Linear inequality b
    eqs_A::Array = SparseMatrixCSC[]                       # Linear equality A vector, in b-Ax=0
    eqs_b::Array = Array[]                                 # Linear equality b
    vks::Array = [Symbol("x",i) for i=1:length(c)]         # Varkeys
    lbs::Dict{Symbol, <:Real} = Dict(vks .=> -Inf)        # Variable lower bounds
    ubs::Dict{Symbol, <:Real} = Dict(vks .=> Inf)         # Variable upper bounds
    int_vks::Array{Symbol} = []                            # Integer variable indices
    JuMP_model::Union{JuMP.Model, Nothing} = nothing       # JuMP model
    JuMP_vars::Union{JuMP.JuMPArray, Nothing} = nothing    # JuMP variables
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

function feasibility(bbf::Union{ModelData, BlackBoxFunction})
    """ Returns the feasibility of data points in a BBF or MD. """
    if isa(bbf, BlackBoxFunction)
        return bbf.feas_ratio
    else
        return [feasibility(fn) for fn in bbf.fns]
    end
end

function accuracy(bbf::Union{ModelData, BlackBoxFunction})
    """ Returns the accuracy of learners in a BBF or MD. """
    if isa(bbf, BlackBoxFunction)
        return bbf.accuracies[end]
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

function add_linear_ineq!(md::ModelData, A::Union{SparseMatrixCSC, Array}, b::Union{Array, <:Real})
    if isa(b, Real)
        @assert size(A, 1) == 1
        push!(md.ineqs_A, A);
        push!(md.ineqs_b, [b]);
    else
        @assert size(A, 1) == size(b, 1)
        push!(md.ineqs_A, A);
        push!(md.ineqs_b, b);
    end
    if !isa(md.JuMP_model, Nothing)
        @constraint(m, md.ineqs_b[end] - md.ineqs_A[end] * x .>= 0);
    end
end

function add_linear_eq!(md::ModelData,  A::Union{SparseMatrixCSC, Array}, b::Union{Array, Real})
    if isa(b, Real)
        @assert size(A, 1) == 1
        push!(md.eqs_A, A);
        push!(md.eqs_b, [b]);
    else
        @assert size(A, 1) == size(b, 1)
        push!(md.eqs_A, A);
        push!(md.eqs_b, b);
    end
    if !isa(md.JuMP_model, Nothing)
        @constraint(m, md.eqs_b[end] - md.eqs_A[end] * x .== 0);
    end
end

function get_max(a, b)
    return maximum([a,b])
end

function get_min(a,b)
    return minimum([a,b])
end

function update_bounds!(md::Union{ModelData, BlackBoxFunction};
                        lbs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}(),
                        ubs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}())
    """ Updates the outer bounds of ModelData and its BlackBoxFunctions, or the bounds
    of a single BlackBoxFunction. """
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
    end
    return
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
    nX = DataFrame();
    if isa(bbf, BlackBoxFunction) && n_comb >= fraction*bbf.n_samples
        @warn("Can't exhaustively sample the boundary of constraint", bbf.name)
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

function knn_sample(bbf::BlackBoxFunction; k::Int64 = 5)
    """ Does KNN and interval arithmetic based sampling once there is at least one feasible
        sample to a BlackBoxFunction. """
    build_knn_tree(bbf);
    idxs, dists = find_knn(bbf, k=k);
    class_dict = classify_patches(bbf, idxs);
    df = reduce(vcat,[secant_method(bbf.X[idx, :], bbf.Y[idx, :]) for idx in class_dict["mixed"]])
    return df
end

function sample_and_eval!(bbf::Union{BlackBoxFunction, ModelData};
                          n_samples:: Union{Int64, Nothing} = nothing,
                          boundary_fraction::Float64 = 0.5,
                          iterations::Int64 = 3,
                          ratio=10.)
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
                             iterations = iterations, ratio = ratio);
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
    else                  # otherwise, KNN sample!
        df = knn_sample(bbf)
        eval!(bbf, df)
    end
    return
end

function add_bounds!(m::JuMP.Model, x::JuMP.JuMPArray, md::Union{ModelData, BlackBoxFunction})
    """Adds outer bounds to JuMP Model from ModelData.lbs/ubs. """
    for vk in md.vks
        if !isinf(md.lbs[vk])
            @constraint(m, x[vk] >= md.lbs[vk])
        end
        if !isinf(md.ubs[vk])
            @constraint(m, x[vk] <= md.ubs[vk])
        end
    end
    return
end

function add_linear_constraints!(m::JuMP.Model, x::JuMP.JuMPArray, md::ModelData)
    """Adds all linear constraints from ModelData into a JuMP.Model."""
    var_list = [x[vk] for vk in md.vks]
    for i=1:length(md.eqs_b)
        @constraint(m, md.eqs_b[i] - md.eqs_A[i] * var_list .== 0);
    end
    for i=1:length(md.ineqs_b)
        @constraint(m, md.ineqs_b[i] .- md.ineqs_A[i] * var_list .>= 0);
    end
    add_bounds!(m, x, md);
    return m
end

function jump_it!(md::ModelData; solver = GurobiSolver())
"""
Creates a JuMP.Model() from the linear constraints of ModelData.
"""
    m = Model(solver=solver);
    @variable(m, x[vk in md.vks])
    if !isempty(md.int_vks)
        aux = @variable(m, [vk in md.int_vks], Int);
        int_vars = [x[vk] for vk in md.int_vks];
        @constraint(m, int_vars .== aux[:]);
    end
    @objective(m, Min, sum(md.c[i] * x[md.vks[i]] for i=1:length(md.vks)));
    add_linear_constraints!(m, x, md);
    md.JuMP_model = m;
    md.JuMP_vars = x;
    return
end

function find_bounds!(md::ModelData; solver=GurobiSolver(), all_bounds=true)

    if isnothing(md.JuMP_model)
        jump_it!(md, solver=solver)
    end
    ubs = Dict(md.vks .=> Inf)
    lbs = Dict(md.vks .=> -Inf)
    # Finding bounds by min/maximizing each variable
    m = md.JuMP_model;
    x = md.JuMP_vars;
    for vk in md.vks
        if isinf(md.lbs[vk]) || all_bounds
            @objective(m, Min, x[vk]);
            status = solve(m);
            lbs[vk] = getvalue(x)[vk];
        end
        if isinf(md.lbs[vk]) || all_bounds
            @objective(m, Max, x[vk]);
            status = solve(m);
            ubs[vk] = getvalue(x)[vk];
        end
    end
    @objective(m, Min, sum(md.c[i] * x[md.vks[i]] for i=1:length(x))); #revert objective
    update_bounds!(md, lbs=lbs, ubs=ubs)
    return
end

function import_trees(dir, md::ModelData)
    """ Returns trees trained over given ModelData,
    where filename points to the model name. """
    trees = [IAI.read_json(string(dir, "_tree_", i, ".json")) for i=1:length(md.fns)];
    return trees
end
using Gurobi
using JuMP
using MathOptInterface
using MosekTools
using LatinHypercubeSampling
using Parameters
using SparseArrays
using Random
Random.seed!(1);

include("exceptions.jl");
include("black_box_function.jl");

@with_kw mutable struct ModelData
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::String = "Model"                                 # Example name
    c::Array                                               # Cost vector
    fns::Array{BlackBoxFn} = Array{BlackBoxFn}[]      # Black box (>/= 0) functions
    ineqs_A::Array = SparseMatrixCSC[]                     # Linear inequality A vector, in b-Ax>=0
    ineqs_b::Array = Array[]                               # Linear inequality b
    eqs_A::Array = SparseMatrixCSC[]                       # Linear equality A vector, in b-Ax=0
    eqs_b::Array = Array[]                                 # Linear equality b
    lbs::Array = -Inf.*ones(length(c))                     # Lower bounds
    ubs::Array = Inf.*ones(length(c))                      # Upper bounds
    int_idxs::Array = []                                   # Integer variable indices
    JuMP_model::Union{JuMP.Model, Nothing} = nothing       # JuMP model
    JuMP_vars::Union{Array, Nothing} = nothing             # JuMP variables
end

function add_fn!(md::ModelData, fn::BlackBoxFn)
    update_bounds!(fn, md.lbs, md.ubs)
    push!(md.fns, fn)
end

function add_linear_ineq!(md::ModelData, A::Union{SparseMatrixCSC, Array}, b::Union{Array, Float64})
    if isa(b, Float64)
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

function add_linear_eq!(md::ModelData,  A::Union{SparseMatrixCSC, Array}, b::Union{Array, Float64})
    if isa(b, Float64)
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

function update_bounds!(md::Union{ModelData, BlackBoxFn}, lbs, ubs)
    """ Updates the outer bounds of ModelData and its BlackBoxFns, or the bounds
    of a single BlackBoxFn. """
    if any(lbs .> ubs)
        throw(ArgumentError("Infeasible bounds."))
    end
    if md.lbs != []
        md.lbs =  [maximum([md.lbs[i], lbs[i]]) for i=1:length(lbs)];
    else
        md.lbs = lbs
    end
    if md.ubs != []
        md.ubs =  [minimum([md.ubs[i], ubs[i]]) for i=1:length(ubs)];
    else
        md.ubs = ubs
    end
    if isa(md, ModelData)
        for fn in md.fns
            update_bounds!(fn, md.lbs, md.ubs);
        end
    end
end

function sample(md::Union{ModelData,BlackBoxFn}; n_samples=1000)
"""
Uniformly samples the variables of ModelData, as long as all
lbs and ubs are defined.
"""
   n_dims = length(md.lbs)
   plan, _ = LHCoptim(n_samples, n_dims, 3);
   if any(isinf.(hcat(md.lbs, md.ubs)))
       throw(ArgumentError("Model is not properly bounded."))
   end
   X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
   return X
end

function add_linear_constraints!(m::JuMP.Model, x::Array{JuMP.Variable}, md::ModelData)
    for i=1:length(md.eqs_b)
        @constraint(m, md.eqs_b[i] - md.eqs_A[i] * x .== 0);
    end
    for i=1:length(md.ineqs_b)
        @constraint(m, md.ineqs_b[i] .- md.ineqs_A[i] * x .>= 0);
    end
    for i=1:length(md.c)
        if !isinf(md.lbs[i])
            @constraint(m, x[i] >= md.lbs[i])
        end
        if !isinf(md.ubs[i])
            @constraint(m, x[i] <= md.ubs[i])
        end
    end
    return m
end

function jump_it!(md::ModelData; solver = GurobiSolver(), trees::Bool = false)
"""
Creates a JuMP.Model() compatible with ModelData.
trees boolean dictates whether tree constraints should be included.
"""
    m = Model(solver=solver);
    @variable(m, x[1:length(md.c)])
    aux = @variable(m, [1:length(md.int_idxs)], Int);
    @constraint(m, x[md.int_idxs] .== aux);
    @objective(m, Min, sum(md.c[i] * x[i] for i=1:length(x)));
    add_linear_constraints!(m, x, md);
    md.JuMP_model = m;
    md.JuMP_vars = x;
    return
end

function find_bounds!(md::ModelData; solver=GurobiSolver(), all_bounds=true)
    if isnothing(md.JuMP_model)
        jump_it!(md, solver=solver, trees = false)
    end
    lbs = -Inf.*ones(length(md.c));
    ubs = Inf.*ones(length(md.c));
    # Finding bounds by min/maximizing each variable
    m = md.JuMP_model;
    x = md.JuMP_vars;
    for i=1:length(md.c)
        if isinf(md.lbs[i]) || all_bounds
            @objective(m, Min, x[i]);
            status = solve(m);
            lbs[i] = getvalue(x)[i];
        end
        if isinf(md.lbs[i]) || all_bounds
            @objective(m, Max, x[i]);
            status = solve(m);
            ubs[i] = getvalue(x)[i];
        end
    end
    @objective(m, Min, sum(md.c[i] * x[i] for i=1:length(x))); #revert objective
    update_bounds!(md, lbs, ubs)
    return
end

function import_trees(dir, md::ModelData)
    """ Returns trees trained over given ModelData,
    where filename points to the model name. """
    trees = [IAI.read_json(string(dir, "_tree_", i, ".json")) for i=1:length(md.fns)];
    return trees
end

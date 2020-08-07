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
    fns::Array{BlackBoxFn} = Array{BlackBoxFn}[]           # Black box (>/= 0) functions
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

function add_fn!(md::ModelData, bbf::BlackBoxFn)
    update_bounds!(bbf, lbs = md.lbs, ubs = md.ubs)
    update_bounds!(md, lbs = bbf.lbs, ubs = bbf.ubs) # TODO: optimize
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

function update_bounds!(md::Union{ModelData, BlackBoxFn};
                        lbs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}(),
                        ubs::Dict{Symbol, <:Real} = Dict{Symbol, Float64}())
    """ Updates the outer bounds of ModelData and its BlackBoxFns, or the bounds
    of a single BlackBoxFn. """
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

function sample(md::Union{ModelData,BlackBoxFn}; n_samples=1000)
"""
Uniformly samples the variables of ModelData, as long as all
lbs and ubs are defined.
"""
   n_dims = length(md.vks)
   plan, _ = LHCoptim(n_samples, n_dims, 3);
   if any(isinf.(values(md.lbs))) || any(isinf.(values(md.ubs)))
       throw(OCTException("Model is not properly bounded."))
   end
   X = scaleLHC(plan,[(md.lbs[vk], md.ubs[vk]) for vk in md.vks]);
   return DataFrame(X, md.vks)
end

function add_linear_constraints!(m::JuMP.Model, x::JuMP.JuMPArray, md::ModelData)
    var_list = [x[vk] for vk in md.vks]
    for i=1:length(md.eqs_b)
        @constraint(m, md.eqs_b[i] - md.eqs_A[i] * var_list .== 0);
    end
    for i=1:length(md.ineqs_b)
        @constraint(m, md.ineqs_b[i] .- md.ineqs_A[i] * var_list .>= 0);
    end
    for vk in md.vks
        if !isinf(md.lbs[vk])
            @constraint(m, x[vk] >= md.lbs[vk])
        end
        if !isinf(md.ubs[vk])
            @constraint(m, x[vk] <= md.ubs[vk])
        end
    end
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
        jump_it!(md, solver=solver, trees = false)
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

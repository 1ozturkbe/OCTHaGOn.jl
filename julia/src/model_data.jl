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

@with_kw mutable struct ModelData
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::String = "Model"                             # Example name
    c::Array                                           # Cost vector
    ineq_fns::Array{Function} = Array{Function}[]      # Inequality (>= 0) functions
    ineq_idxs::Array{Union{Nothing, Array}} = Array[]  # Inequality function variable indices
    eq_fns::Array{Function} = Array{Function}[]        # Equality (>= 0) functions
    eq_idxs::Array{Union{Nothing, Array}} = Array[]    # Equality function variable indices
    ineqs_A::Array = SparseMatrixCSC[]                 # Linear inequality A vector, in b-Ax>=0
    ineqs_b::Array = Array[]                           # Linear inequality b
    eqs_A::Array = SparseMatrixCSC[]                   # Linear equality A vector, in b-Ax=0
    eqs_b::Array = Array[]                             # Linear equality b
    lbs::Array = -Inf.*ones(length(c))                 # Lower bounds
    ubs::Array = Inf.*ones(length(c))                  # Upper bounds
    convex_idxs::Array = []                            # Convex inequality indices
    logconvex_idxs::Array = []                         # Log-convex inequality indices
    int_idxs::Array = []                               # Integer variable indices
end

function add_ineq_fn!(md::ModelData, fn::Function, idxs::Union{Nothing, Array})
    push!(md.ineq_fns, fn)
    push!(md.ineq_idxs, idxs)
end

function add_eq_fn!(md::ModelData, fn::Function, idxs::Union{Nothing, Array})
    push!(md.eq_fns, fn)
    push!(md.eq_idxs, idxs)
end

function update_bounds!(md::ModelData, lbs, ubs)
    if any(lbs .> ubs)
        throw(ArgumentError("Infeasible bounds."))
    end
    md.lbs =  [maximum([md.lbs[i], lbs[i]]) for i=1:length(md.c)];
    md.ubs =  [minimum([md.ubs[i], ubs[i]]) for i=1:length(md.c)];
end

# function find_bounds!(md::ModelData; solver=Mosek.Optimizer(), feasible = zeros(length(md.c)),
#                         n_samples=100)
# """
# Finds the missing md.lbs and md.ubs using rapidly exploring random trees.
# """
#     n_dims = length(md.c);
#     upper_unbounded = isinf.(md.ubs);
#     lower_unbounded = isinf.(md.lbs);
#     lbs = deepcopy(md.lbs);
#     ubs = deepcopy(md.ubs);
#     # Will sample single-tailed in logspace
#     plan, _ = LHCoptim(n_samples, n_dims, 3);
#     log_jumps = scaleLHC(plan, [(-3, 10) for i=1:n_dims])
#     X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
#     feas_matrix = ones(n_samples, length(md.))
#     for i = 1:n_dims
#         np =
#     m = Model(solver=solver);
# end

function sample(md::ModelData; n_samples=1000)
"""
Samples the variables of ModelData, as long as all
lbs and ubs are defined.
"""
   n_dims = length(md.c);
   plan, _ = LHCoptim(n_samples, n_dims, 3);
   if any(isinf.(hcat(md.lbs, md.ubs)))
       throw(ArgumentError("Model is not properly bounded."))
   end
   X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
   return X
end

function add_linear_constraints!(m::JuMP.Model, x::Array{JuMP.Variable}, md::ModelData)
    for i=1:length(md.eqs_b)
        terms = length(md.eqs_b[i]);
        @constraint(m, md.eqs_b[i] .- [sum(md.eqs_A[i][j,:] .* x) for j=1:terms] .== 0);
    end
    for i=1:length(md.ineqs_b)
        terms = length(md.ineqs_b[i]);
        @constraint(m, md.ineqs_b[i] .- [sum(md.ineqs_A[i][j,:] .* x) for j=1:terms] .>= 0);
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

function jump_it(md::ModelData; solver = GurobiSolver())
"""
Creates a JuMP.Model() compatible with ModelData,
with only the linear constraints.
"""
    m = Model(solver=solver);
    @variable(m, x[1:length(md.c)])
    aux = @variable(m, [1:length(md.int_idxs)], Int);
    @constraint(m, x[md.int_idxs] .== aux);
    @objective(m, Min, sum(md.c .* x));
    add_linear_constraints!(m, x, md);
    return m,x
end

function find_bounds!(md::ModelData; solver=GurobiSolver(), all_bounds=true)
    m, x = jump_it(md, solver=solver)
    lbs = -Inf.*ones(length(md.c));
    ubs = Inf.*ones(length(md.c));
    # Finding bounds by min/maximizing each variable
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
    update_bounds!(md, lbs, ubs)
end

function import_trees(dir, md::ModelData)
    """ Returns trees trained over given ModelData,
    where filename points to the model name. """
    ineq_trees = [IAI.read_json(string(dir, "_ineq_", i, ".json")) for i=1:length(md.ineq_fns)];
    eq_trees = [IAI.read_json(string(dir, "_eq_", i, ".json")) for i=1:length(md.eq_fns)];
    return ineq_trees, eq_trees
end

function show_trees(trees)
    """ Shows all trees (grids) in browser. """
    for tree in trees
        try
            IAI.show_in_browser(tree.lnr)
        catch err
            if isa(err, UndefRefError)
            @warn("Certain trees are untrained.")
            else
                rethrow(err)
            end
        end
    end
end
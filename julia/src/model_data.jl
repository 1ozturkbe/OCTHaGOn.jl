using Gurobi
using JuMP
using LatinHypercubeSampling
using Parameters
using SparseArrays

@with_kw mutable struct ModelData
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::String = "Model"                          # Example name
    c::Array                                        # Cost vector
    ineq_fns::Array{Function} = Array{Function}[]   # Inequality (>= 0) functions
    ineq_idxs::Array = Array[]                      # Inequality function variable indices
    eq_fns::Array{Function} = Array{Function}[]     # Equality (>= 0) functions
    eq_idxs::Array = Array[]                        # Equality function variable indices
    ineqs_A::Array = SparseMatrixCSC[]              # Linear inequality A vector, in b-Ax>=0
    ineqs_b::Array = []                             # Linear inequality b
    eqs_A::Array = SparseMatrixCSC[]                # Linear equality A vector, in b-Ax=0
    eqs_b::Array = []                               # Linear equality b
    lbs::Array = -Inf.*ones(length(c))              # Lower bounds
    ubs::Array = Inf.*ones(length(c))               # Upper bounds
    int_idxs::Array = []                            # Integer variable indices
end

function update_bounds!(md::ModelData, lbs, ubs)
    if any(lbs .> ubs)
        throw(ArgumentError("Infeasible bounds."))
    end
    md.lbs =  [maximum([md.lbs[i], lbs[i]]) for i=1:length(md.c)];
    md.ubs =  [minimum([md.ubs[i], ubs[i]]) for i=1:length(md.c)];
end

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

function jump_it(md::ModelData; solver = GurobiSolver())
"""
Creates a JuMP.Model() compatible with ModelData,
with only the linear constraints.
"""
    n_vars = length(md.c);
    m = Model(solver=solver);
    @variable(m, x[1:n_vars]);
    @objective(m, Min, sum(md.c.*x));
    return m, x
end

function import_trees(dir, md::ModelData)
    """ Returns trees trained over given ModelData,
    where filename points to the model name. """
    ineq_trees = [IAI.read_json(string(dir, "_ineq_", i, ".json")) for i=1:length(md.ineq_fns)];
    eq_trees = [IAI.read_json(string(dir, "_eq_", i, ".json")) for i=1:length(md.eq_fns)];
    return ineq_trees, eq_trees
end

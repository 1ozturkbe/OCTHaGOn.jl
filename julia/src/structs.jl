using Parameters

@with_kw mutable struct ModelData
    name::String = "Model"                        # Example name
    c::Array                                      # Cost vector
    ineq_fns::Array{Function} = Array{Function}[] # Inequality (>= 0) functions
    ineq_idxs::Array = Array[]                    # Inequality function indices
    eq_fns::Array{Function} = Array{Function}[]   # Equality (>= 0) functions
    eq_idxs::Array = Array[]                      # Equality function indices
    ineqs_A::Array = Array[]                      # Linear inequality A vector, in b-Ax>=0
    ineqs_b::Array{Float64} = []                  # Linear inequality b
    eqs_A::Array = Array[]                        # Linear equality A vector, in b-Ax=0
    eqs_b::Array{Float64} = []                    # Linear equality b
    lbs::Array = -Inf.*ones(length(c))            # Lower bounds
    ubs::Array = Inf.*ones(length(c))             # Upper bounds
end

function update_bounds!(md::ModelData, lbs, ubs)
    if any(lbs .>= ubs)
        throw(ArgumentError("Infeasible bounds."))
    end
    md.lbs =  [maximum([md.lbs[i], lbs[i]]) for i=1:length(md.c)];
    md.ubs =  [minimum([md.ubs[i], ubs[i]]) for i=1:length(md.c)];
end
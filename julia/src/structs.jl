using Parameters

@with_kw mutable struct ModelData
    name::String = "Model"                        # Example name
    c::Array                                      # Cost vector
    ineq_fns::Array{Function} = Array{Function}[] # Inequality (>= 0) functions
    ineq_idxs::Array = Array[]                    # Inequality function indices
    eq_fns::Array{Function} = Array{Function}[]   # Equality (>= 0) functions
    eq_idxs::Array = Array[]                      # Equality function indices
    eqs_A::Array = Array[]                        # Linear equality A vector, in b-Ax=0
    eqs_b::Array{Float64} = []                    # Linear equality b
    lbs::Array = -Inf.*ones(length(c))            # Lower bounds
    ubs::Array = Inf.*ones(length(c))             # Upper bounds
end

struct constraint_function
end
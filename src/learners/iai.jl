@with_kw mutable struct OCTH_Classifier
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT

    # Output
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
    
    # Methods
    fit!::Function = svm_fit
    predict::Function = svm_cl_predict
end

@with_kw mutable struct OCTH_Regressor
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT

    # Output
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
    
    # Methods
    fit!::Function = svm_fit
    predict::Function = svm_r_predict
end
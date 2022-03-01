
@with_kw mutable struct SVM_Classifier <: AbstractModel
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT

    # Output
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
    
    # Methods
    fit!::Function = svm_fit
    predict::Function = svm_cl_predict
    embed_mio!::Function = svm_cl_embed
end

@with_kw mutable struct SVM_Regressor <: AbstractModel
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT

    # Output
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
    
    # Methods
    fit!::Function = svm_fit
    predict::Function = svm_r_predict
    embed_mio!::Function = svm_r_embed
end

"""
Used to fit an SVM classifier/regressor
"""
function svm_fit(lnr::SVM_Classifier, X::DataFrame, Y::Array)

    X = Matrix(X)

    solver = lnr.solver

    m = JuMP.Model(with_optimizer(solver))
    @variable(m, z[1:length(Y)] >= 0)
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    for i=1:length(Y)
        @constraint(m,  z[i] >=  Y[i] - β0 - sum(X[i,:] .* β))
        @constraint(m,  z[i] >= -Y[i] + β0 + sum(X[i,:] .* β))
    end
    @objective(m, Min, lnr.C*sum(z)+β'*β)
    optimize!(m)

    lnr.β0 = JuMP.value(β0)
    lnr.β = JuMP.value.(β)
end

"""
Used to make predictions based on a learned SVM classifier
"""
function svm_cl_predict(lnr::SVM_Classifier, X::DataFrame; continuous=false)

    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    X = Matrix(X)

    logit = lnr.β0 .+ X * lnr.β
    if !continuous
        logit = 1*(logit .>= 0.5)
    end

    return logit
end

"""
Used to make predictions based on a learned SVM regressor
"""
function svm_r_predict(lnr::SVM_Regressor, X::DataFrame)
    
    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    X = Matrix(X)
    return lnr.β0 .+ X * lnr.β
end

"""
Embed MIO constraints on SVM classifier
"""
function svm_cl_embed(lnr::SVM_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)
    
    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    X = Matrix(bbl.X)

    m, x = gm.model, bbl.vars

    β0, β = lnr.β0, lnr.β
    
    con = @constraint(m, x'*β .+ β0 .+ 0.5 >=0)

    return Dict(1 => [con]), Dict()

end

"""
Embed MIO constraints on SVM regressor
"""
function svm_r_embed(lnr::SVM_Regressor, m::JuMP.Model, x::Array{JuMP.VariableRef}; kwargs...)

    error("Not implemented yet")
end
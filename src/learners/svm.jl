
@with_kw mutable struct SVM_Classifier <: AbstractClassifier
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT
    equality::Bool = false # Whether or not we are dealing with an equality constraint 
    use_epsilon::Bool = false # Whether we should round Y using EPSILON tolerance in case of equality
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing

    # Model variables
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
    thres::Real = 0.5

end

@with_kw mutable struct SVM_Regressor <: AbstractRegressor
    # Arguments
    C::Real = 1
    solver = CPLEX_SILENT
    equality::Bool = false
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing
    
    # Model variables
    β0::Union{Nothing, Real} = nothing
    β::Union{Nothing, Vector{Float64}} = nothing
end

function convert_to_binary(lnr::SVM_Classifier, Y::Array)
    return (lnr.equality && lnr.use_epsilon ? 1*(abs.(Y .- lnr.thres) .<= EPSILON) : 1*(Y .>= lnr.thres));
end

"""
Used to fit an SVM classifier
"""
function fit!(lnr::SVM_Classifier, X::DataFrame, Y::Array; equality::Bool=false)
    lnr.equality = equality

    Y_hat = 1*(Y .>= 0) 

    if equality
        tmp = abs.(Y) .<= EPSILON
        positive_sample_fraction = sum(tmp) / length(Y);
        if positive_sample_fraction >= 0.1
            Y_hat = tmp
            lnr.use_epsilon = true
        else 
            # In this case, we will continue modeling as inequality instead of equality
            println("Not enough samples to SVM approximate equality constraint: $(positive_sample_fraction)")
        end
    end

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
Used to fit an SVM regressor
"""
function fit!(lnr::SVM_Regressor, X::DataFrame, Y::Array; equality::Bool=false)
    lnr.equality = equality

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
function predict(lnr::SVM_Classifier, X::DataFrame; continuous=false)

    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    X = Matrix(X)

    logit = lnr.β0 .+ X * lnr.β
    if !continuous
        logit = convert_to_binary(lnr, logit)
    end

    return logit
end

"""
Used to make predictions based on a learned SVM regressor
"""
function predict(lnr::SVM_Regressor, X::DataFrame)
    
    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    X = Matrix(X)
    return lnr.β0 .+ X * lnr.β
end

"""
Evaluate using SVM in classification task
"""
function evaluate(lnr::SVM_Classifier, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = classification_evaluation()

    score = evaluator.second(y_pred, convert_to_binary(lnr, Y))
    return score
end

"""
Evaluate using SVM in regression task
"""
function evaluate(lnr::SVM_Regressor, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = regression_evaluation()

    score = evaluator.second(y_pred, Y)
    return score
end



"""
Embed MIO constraints on SVM classifier
"""
function embed_mio!(lnr::SVM_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)
    
    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    ro_factor = get_param(gm, :ro_factor);

    m, x = gm.model, bbl.vars

    β0, β = lnr.β0, lnr.β
    
    cons = []

    if ro_factor == 0
        push!(cons, @constraint(m, x'*β + β0 + lnr.thres >=0))
    else
        P = ro_factor*diagm(1.0*β)
        
        # Create variables that will be used for robustness
        var_name = eval(Meta.parse(":t_rsvm_$(bbl.name)"));
        m[var_name] = @variable(m, base_name=string(var_name));
        t_var = m[var_name];

        push!(cons, @constraint(m, x'*β + β0 + lnr.thres - t_var >=0))
        append!(cons, @constraint(m, P*x .<= t_var))
        append!(cons, @constraint(m, -P*x .<= t_var))

    end
    return Dict(1 => cons), Dict()

end

"""
Embed MIO constraints on SVM regressor
"""
function embed_mio!(lnr::SVM_Regressor, gm::GlobalModel, bbl::BlackBoxRegressor; kwargs...)

    if isnothing(lnr.β0)
        error("SVM model hasn't been fitted")
    end

    m, x = gm.model, bbl.vars

    β0, β = lnr.β0, lnr.β
    
    cons = []

    if !isnothing(lnr.dependent_var) && lnr.equality
        push!(cons, @constraint(m, x'*β .+ β0 == lnr.dependent_var))
    elseif lnr.equality
        push!(cons, @constraint(m, x'*β .+ β0 >= -EPSILON))
        push!(cons, @constraint(m, x'*β .+ β0 <= EPSILON))
    else 
        push!(cons, @constraint(m, x'*β .+ β0 >= 0))
    end
    # println("Doing whatever")
    return Dict(1 => cons), Dict()
end
using StatsBase: sample, countmap

@with_kw mutable struct MPC_Classifier <: AbstractClassifier
    # Arguments
    K::Integer = 5
    L::Integer = 3
    MAX_ITER::Integer = 40
    TIME_LIMIT::Real = 50
    TOTAL_TIME_LIMIT::Real = 100
    #solver = CPLEX_SILENT
    equality::Bool = false # Whether or not we are dealing with an equality constraint 
    use_epsilon::Bool = false # Whether we should round Y using EPSILON tolerance in case of equality
    dependent_var::Union{Nothing, JuMP.VariableRef} = nothing

    # Model variables
    A::Union{Nothing, Array} = nothing
    b::Union{Nothing, Array} = nothing

    best_val_accuracy::Float64 = 0
    best_A::Union{Nothing, Array} = nothing
    best_b::Union{Nothing, Array} = nothing

end

# @with_kw mutable struct MPC_Regressor <: AbstractRegressor
#     # Arguments
#     K::Integer = 1
#     L::Integer = 1
#     MAX_ITER::Integer = 20
#     TIME_LIMIT::Real = 100
#     TOTAL_TIME_LIMIT::Real = 200
#     #solver = CPLEX_SILENT
#     equality::Bool = false # Whether or not we are dealing with an equality constraint 
#     use_epsilon::Bool = false # Whether we should round Y using EPSILON tolerance in case of equality
#     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing

#     # Model variables
#     A::Union{Nothing, Array} = nothing
#     b::Union{Nothing, Array} = nothing
    
#     best_val_accuracy::Float64 = 0
#     best_A::Union{Nothing, Array} = nothing
#     best_b::Union{Nothing, Array} = nothing
# end

function accuracy_score(y_true::Vector, y_pred::Vector)
    if length(y_true) != length(y_pred)
        error("y_true and y_pred must have the same length")
    end
    n_correct = count(y_true .== y_pred)
    return n_correct / length(y_true)
end

function train_test_split(X::Matrix, Y::Union{Array, Vector, BitVector}; at = 0.7)
    n = size(X, 1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    (X[train_idx,:], Y[train_idx]), (X[test_idx,:], Y[test_idx])
end

function stratified_train_test_split(x::Matrix, y::Union{Array, Vector, BitVector}; at = 0.7, random_state=42)
    """
    Split the input data into training and testing sets using stratified sampling.

    Parameters:
    - x: Input features as an array.
    - y: Input labels as an array.
    - test_size: The fraction of the data to reserve for testing. Default is 0.2.
    - random_state: The seed for the random number generator. Default is 1234.

    Returns:
    - train_x: The training input features.
    - test_x: The testing input features.
    - train_y: The training input labels.
    - test_y: The testing input labels.
    """
    test_size = 1-at
    n = length(y)
    n_test = Int(round(n * test_size))
    n_train = n - n_test

    # Determine the unique classes in the target variable
    classes = unique(y)

    # Compute the number of samples in each class
    class_counts = countmap(y)
    rng = MersenneTwister(random_state)
    # Compute the number of samples to select from each class for the test set
    #class_test_counts = round.(Int, class_counts .* test_size)
    class_test_counts = Dict((a+1,round(Int, b*test_size)) for (a,b) in countmap(y))

    class_indices = Dict(cls=>findall(y.==cls-1) for cls=1:length(classes))
    #class_data = [x[class_indices[i], :] for i in 1:length(classes)]
    #class_labels = [y[class_indices[i], :] for i in 1:length(classes)]

    # Select samples for the test set
    test_indices = []
    for i in 1:length(classes)
        test_indices = vcat(test_indices, sample(rng, class_indices[i], class_test_counts[i], replace=false))
    end
    test_x = x[test_indices, :]
    test_y = y[test_indices]

    # Select samples for the train set
    train_indices = setdiff(1:n, test_indices)
    train_x = x[train_indices, :]
    train_y = y[train_indices]

    return (train_x,train_y), (test_x, test_y)
end

function MPCM_solve(X, y, K, L, time_limit)
    
    N, M = size(X)
    eps_A = 0.1
    eps_b = 0.1
    eps = 0.0001
    eps_M = 1
    big_M2 = eps_M + (M + eps_A) + (eps_b)
    big_M1 = eps_M + (M + eps_A) + (1 + eps_b) + eps

    Y0 = findall(y .== 0)
    Y1 = findall(y .== 1)
    #println("Y0_len $(length(Y0))")
    #println("Y1_len $(length(Y1))")
    
    model = Model(with_optimizer(Gurobi.Optimizer))
    #@variable(model, x[1:100])
    @variable(model, A[1:L, 1:K, 1:M], lower_bound =-(1+eps_A), upper_bound=(1+eps_A))
    @variable(model, b[1:L, 1:K], lower_bound =-eps_b, upper_bound=1+eps_b)
    @variable(model, z[1:L, 1:K, 1:N], Bin)
    @variable(model, p[Y1, 1:L], Bin)
    @variable(model, pp[Y1], Bin)
    @variable(model, n[Y0, 1:L], Bin)

    @objective(model, Min, sum(n[i, l] for i in Y0 for l in 1:L) + sum(pp[i] for i in Y1))

    @constraint(model, [i=1:N, k=1:K, l=1:L], 
                sum(A[l, k, j] * X[i, j] for j in 1:M) >= b[l, k] + eps - big_M1 * (1-z[l, k, i]))

    @constraint(model, [i=1:N, k=1:K, l=1:L], 
                sum(A[l, k, j] * X[i, j] for j in 1:M) <= b[l, k] + big_M2 * z[l, k, i])

    @constraint(model, [i=Y1, l=1:L], 
                K * p[i, l] >= sum(z[l, k, i] for k in 1:K))

    @constraint(model, [i=Y0, l=1:L], 
                n[i, l] >= 1 - sum(z[l, k, i] for k in 1:K))

    @constraint(model, [i=Y1], 
                pp[i] >= sum(p[i, l] for l in 1:L) - L + 1)
    #display(model)
    println("start optimization")
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    optimize!(model)

    return JuMP.value.(A), JuMP.value.(b), model
end

function convert_to_binary(lnr::MPC_Classifier, Y::Array)
    #return (lnr.equality && lnr.use_epsilon ? 1*(abs.(Y .- lnr.thres) .<= EPSILON) : 1*(Y .>= lnr.thres));
    return 1*(Y .>= 0);
end

"""
Used to fit an SVM classifier
"""
function fit!(lnr::MPC_Classifier, X::DataFrame, y::Array; equality::Bool=false)
    MAX_ITER = 20
    
    X = Matrix(X)
    
    Y_hat = 1*(y .>= 0)

    # Split into training/validation set (validation is used to pick best iteration)
    (X, y), (X_val, y_val) = stratified_train_test_split(X, Y_hat; at=0.8)

    K = lnr.K
    L = lnr.L
    #K,L = 10, 1 
    
    Random.seed!(0)

    

    n_samples = 10
    idx = Set(sample(1:size(X, 1), n_samples, replace=false))

    X_0 = X[collect(idx), :]
    y_0 = y[collect(idx)]

    it = 1

    prev_idx_len = -1
    start_time = time()

    while true
        println(" ################## Iteration $it ################## ")
        println("==================[Number of points=$(length(idx))]====================")
        #println("==================[Number of points2=$(length(y_0))]====================")
        A, b, model = MPCM_solve(X_0, y_0, K, L, lnr.TIME_LIMIT)

        #lnr.A, lnr.b = convert(Array{Float64,2}, A), convert(Array{Float64,1}, b)
        lnr.A, lnr.b = A, b
        
        y_pred = predict(lnr, X)

        idx_misclass = Set(findall(y_pred .!= y))

        accuracy_train = accuracy_score(y, y_pred)
        accuracy_val = accuracy_score(y_val, predict(lnr, X_val))
        println("#########################################################")
        println(" - Training Accuracy: $accuracy_train")
        println(" - Validation Accuracy: $accuracy_val")
        println(" - Number of new misclassified samples: $(length(idx_misclass))")
        println("#########################################################")

        if accuracy_val >= lnr.best_val_accuracy
            lnr.best_val_accuracy = accuracy_val
            lnr.best_A = copy(lnr.A)
            lnr.best_b = copy(lnr.b)
        end

        u = 10
        if length(idx_misclass) > u
            idx_misclass = Set(sample(collect(idx_misclass), min(u, length(idx_misclass)), replace=false))
        end

        if length(idx_misclass) > 0
            idx = idx ∪ idx_misclass
            X_0 = X[collect(idx), :]
            y_0 = y[collect(idx)]
            it += 1
        else
            break
        end

        # Different termination conditions:
        # 1) Number of iterations exceeds a maximum number
        # 2) the set of sample indexes remains the same
        # 3) the time exceeds a total time threshold set by the user
        if it > MAX_ITER || prev_idx_len == length(idx) || time() - start_time > lnr.TOTAL_TIME_LIMIT
            @show prev_idx_len == length(idx)
            break
        end
        prev_idx_len = length(idx)
    end

    lnr.A = copy(lnr.best_A)
    lnr.b = copy(lnr.best_b)
    
    return lnr.A, lnr.b
end

"""
Used to fit an SVM regressor
"""
# function fit!(lnr::MPC_Regressor, X::DataFrame, Y::Array; equality::Bool=false)
#     lnr.equality = equality

#     X = Matrix(X)

#     solver = lnr.solver

#     m = JuMP.Model(with_optimizer(solver))
#     @variable(m, z[1:length(Y)] >= 0)
#     @variable(m, β[1:size(X, 2)])
#     @variable(m, β0)
#     for i=1:length(Y)
#         @constraint(m,  z[i] >=  Y[i] - β0 - sum(X[i,:] .* β))
#         @constraint(m,  z[i] >= -Y[i] + β0 + sum(X[i,:] .* β))
#     end
#     @objective(m, Min, lnr.C*sum(z)+β'*β)
#     optimize!(m)

#     lnr.β0 = JuMP.value(β0)
#     lnr.β = JuMP.value.(β)
# end

"""
Used to make predictions based on a learned SVM classifier
"""
function predict(lnr::MPC_Classifier, X::Matrix; continuous=false)

    A, b = lnr.A, lnr.b
    polyhedra_preds = [all(X * lnr.A[i,:,:]' .<= lnr.b[i,:]', dims=2)[:,1] for i=1:size(lnr.A, 1)]
    preds = 1*any(permutedims(hcat(polyhedra_preds...)), dims=1)[1,:]
    return preds
end

"""
Used to make predictions based on a learned SVM classifier
"""
function predict(lnr::MPC_Classifier, X::DataFrame; continuous=false)

    return predict(lnr, Matrix(X))
end




"""
Used to make predictions based on a learned SVM regressor
"""
# function predict(lnr::MPC_Regressor, X::DataFrame)
    
#     if isnothing(lnr.β0)
#         error("SVM model hasn't been fitted")
#     end

#     X = Matrix(X)
#     return lnr.β0 .+ X * lnr.β
# end

"""
Evaluate using SVM in classification task
"""
function evaluate(lnr::MPC_Classifier, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = classification_evaluation()

    score = evaluator.second(y_pred, convert_to_binary(lnr, Y))
    return score
end

"""
Evaluate using SVM in regression task
"""
# function evaluate(lnr::MPC_Regressor, X::DataFrame, Y::Array)
    
#     y_pred = predict(lnr, X)

#     evaluator = regression_evaluation()

#     score = evaluator.second(y_pred, Y)
#     return score
# end



"""
Embed MIO constraints on SVM classifier
"""
function embed_mio!(lnr::MPC_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)
    
    cons = []
    
    bigM = 10

    L, K, M = size(lnr.A)

    m, x = gm.model, bbl.vars
    
    #@variable(m, x[1:2]);
    zz = @variable(m, [1:L], Bin);
    
    #append!(cons, @constraint(m,[p=1:L], lnr.A[p,:,:]*x .<= lnr.b[p] + bigM*(1-zz[p])))
    #push!(cons, @constraint(m, sum(zz)>=1))
    
    push!(cons, @constraint(m, sum(zz)>=1))
    for p = 1:L
        cons = vcat([cons, @constraint(m, lnr.A[p,:,:]*x .<= lnr.b[p] + bigM*(1-zz[p]))]...)
    end
    
    return Dict(1 => cons), Dict()
end

"""
Embed MIO constraints on SVM regressor
"""
# function embed_mio!(lnr::MPC_Regressor, gm::GlobalModel, bbl::BlackBoxRegressor; kwargs...)

#     if isnothing(lnr.β0)
#         error("SVM model hasn't been fitted")
#     end

#     m, x = gm.model, bbl.vars

#     β0, β = lnr.β0, lnr.β
    
#     cons = []

#     if !isnothing(lnr.dependent_var) && lnr.equality
#         push!(cons, @constraint(m, x'*β .+ β0 == lnr.dependent_var))
#     elseif lnr.equality
#         push!(cons, @constraint(m, x'*β .+ β0 >= -EPSILON))
#         push!(cons, @constraint(m, x'*β .+ β0 <= EPSILON))
#     else 
#         push!(cons, @constraint(m, x'*β .+ β0 >= 0))
#     end
#     # println("Doing whatever")
#     return Dict(1 => cons), Dict()
# end


@with_kw mutable struct MLP_Regressor <: AbstractRegressor
    # Arguments
    solver = CPLEX_SILENT
    equality::Bool = false

    # Model variables
    mlp::Union{MLP, Nothing} = nothing
end

@with_kw mutable struct MLP_Classifier <: AbstractClassifier
    # Arguments
    solver = CPLEX_SILENT
    equality::Bool = false

    # Model variables
    mlp::Union{MLP, Nothing} = nothing

end

"""
Used to fit an MLP regressor
"""
function fit!(lnr::Union{MLP_Regressor, MLP_Classifier}, X::DataFrame, Y::Array; equality=false)


    X = Matrix(X)
    layer_sizes = [size(X,2), 100, 1]

    act = Vector{Function}([relu,  logis])
    actd = Vector{Function}([relud, logisd])

    # initialize net
    lnr.mlp = MLP(randn, layer_sizes, act, actd)

    #println(Matrix(Y'))
    gdmtrain(lnr.mlp, Matrix(X'), Matrix(1*(Y .>= 0)'); batch_size=16,learning_rate=0.01,
        momentum_rate=0.01,maxiter=20, show_trace=true);

end


"""
Used to make predictions based on a learned MLP regressor
"""
function predict(lnr::MLP_Classifier, X::DataFrame)
    
    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    X = Matrix(Matrix(X)')
    return 1*(prop(lnr.mlp, X) .>= 0.5)
end


"""
Used to make predictions based on a learned MLP regressor
"""
function predict(lnr::MLP_Regressor, X::DataFrame)
    
    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    X = Matrix(Matrix(X)')
    return prop(lnr.mlp, X)
end


"""
Evaluate using MLP in regression task
"""
function evaluate(lnr::MLP_Classifier, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = classification_evaluation()

    score = evaluator.second(y_pred, 1*(Y .>= 0))
    return score
end
"""
Evaluate using MLP in regression task
"""
function evaluate(lnr::MLP_Regressor, X::DataFrame, Y::Array)
    
    y_pred = predict(lnr, X)

    evaluator = regression_evaluation()

    score = evaluator.second(y_pred, Y)
    return score
end



"""
Embed MIO constraints on MLP regressor
"""
function embed_mio!(lnr::MLP_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)

    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    mlp = lnr.mlp
    m, x = gm.model, bbl.vars
    
    cons = []

    M_u, M_l = 1e5, -1e5

    layer_input = x

    # @TODO: probably use dependent var here?
    var_name = eval(Meta.parse(":y_nn_$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    y = m[var_name];

    max_layers = length(mlp.net)


    for (i, layer) in enumerate(mlp.net)
        W, b = layer.w, layer.b
        if i == max_layers
            @constraint(m, y .== layer_input'*W[1, :] + b)
        else 
            v_pos_list = []
            
            for node_id in 1:size(W, 1)
                v = @variable(m, base_name="v_$(i)_$(node_id)")
                v_ind = @variable(m, base_name="v_ind_$(i)_$(node_id)")
                push!(v_pos_list, v)

                append!(cons, [
                    @constraint(m, v >= 0),
                    @constraint(m, v >= W[node_id, :]'*layer_input + b[node_id]),
                    @constraint(m, v <= M_u*v_ind),
                    @constraint(m, v <= W[node_id, :]'*layer_input + b[node_id]-M_l*v_ind)
                ])
                
            end
            layer_input = v_pos_list
        end
    end
    push!(cons, 
        @constraint(m, y>= 0)
    )
    
    return Dict(1 => cons), Dict()
end



"""
Embed MIO constraints on MLP regressor
"""
function embed_mio!(lnr::MLP_Regressor, gm::GlobalModel, bbl::BlackBoxRegressor; kwargs...)

    if isnothing(lnr.mlp)
        error("MLP model hasn't been fitted")
    end

    mlp = lnr.mlp
    m, x = gm.model, bbl.vars
    
    cons = []

    M_u, M_l = 1e5, -1e5

    layer_input = x

    # @TODO: probably use dependent var here?
    var_name = eval(Meta.parse(":y_nn_$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    y = m[var_name];

    max_layers = length(mlp.net)


    for (i, layer) in enumerate(mlp.net)
        W, b = layer.w, layer.b
        if i == max_layers
            @constraint(m, y .== layer_input'*W[1, :] + b)
        else 
            v_pos_list = []
            
            for node_id in 1:size(W, 1)
                v = @variable(m, base_name="v_$(i)_$(node_id)")
                v_ind = @variable(m, base_name="v_ind_$(i)_$(node_id)")
                push!(v_pos_list, v)

                append!(cons, [
                    @constraint(m, v >= 0),
                    @constraint(m, v >= W[node_id, :]'*layer_input + b[node_id]),
                    @constraint(m, v <= M_u*v_ind),
                    @constraint(m, v <= W[node_id, :]'*layer_input + b[node_id]-M_l*v_ind)
                ])
                
            end
            layer_input = v_pos_list
        end
    end
    push!(cons, 
        @constraint(m, y>= 0)
    )
    
    return Dict(1 => cons), Dict()
end
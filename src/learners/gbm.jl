using JLBoost

EPSILON = 0.5

abstract type AbstractGBM <: AbstractModel end

@with_kw mutable struct GBM_Classifier <: AbstractGBM
    # Arguments
    max_depth::Int64 = 6
    solver = CPLEX_SILENT

    # Model data
    gbm::Union{Nothing, JLBoostTreeModel} = nothing
    
    # Methods
    fit!::Function = gbm_cl_fit
    predict::Function = gbm_cl_predict
    embed_mio!::Function = gbm_cl_embed
end

@with_kw mutable struct GBM_Regressor <: AbstractGBM
    # Arguments
    max_depth::Int64 = 6
    solver = CPLEX_SILENT

    # Model data
    gbm::Union{Nothing, JLBoostTreeModel} = nothing
    
    # Methods
    fit!::Function = gbm_r_fit
    predict::Function = gbm_predict
    embed_mio!::Function = gbm_cl_embed
end


"""
Fit gbm in classification task
"""
function gbm_cl_fit(lnr::GBM_Classifier , X::DataFrame, Y::Array; equality=false)
    
    df = deepcopy(X)

    Y_hat = 1*(Y .>= 0) 
    
    if equality
        tmp = abs.(Y) .<= EPSILON
        positive_sample_fraction = sum(tmp)/length(Y);
        if positive_sample_fraction >= 0.1
            Y_hat = tmp
        else 
            # In this case, we will continue modeling as inequality instead of equality
            println("Not enough samples to GBM approximate equality constraint: $(positive_sample_fraction)")
        end
    end
    df[!,"output"] = Y_hat;
    lnr.gbm = jlboost(df, "output"; verbose=false, max_depth = lnr.max_depth);
end

function gbm_r_fit(lnr::GBM_Regressor , X::DataFrame, Y::Array; equality=false)
    
    df = deepcopy(X)
    df[!,"output"] = Y;
    lnr.gbm = jlboost(df, "output"; verbose=false, max_depth = lnr.max_depth);
end

function gbm_cl_predict(lnr::GBM_Classifier, X::DataFrame; continuous=false)
    
    if isnothing(lnr.gbm)
        error("GBM model hasn't been fitted yet")
    end
    logits = JLBoost.predict(lnr.gbm, X)
    if !continuous
        logits = 1*(logits .>= 0.5)
    end
    return logits
end

function find_all_leaves(tree::AbstractJLBoostTree)
    if length(tree.children) == 0
        return [tree]
    else
        leaves = []
        for child in tree.children
            append!(leaves, find_all_leaves(child))
        end
        return leaves
    end
end

function update_leaf_df!(df::DataFrame, leaf::AbstractJLBoostTree, cols::Vector{String}, leaf_id::Int64, leaf_weight::Float64)
    
    if isnothing(leaf.parent)
        return
    end
    
    # Initialize all variables to 0
    d = Dict(c => 0.0 for c in cols)
    
    # Change value of splitting feature
    d[String(leaf.parent.splitfeature)] = (leaf == leaf.parent.children[1] ? 1 : -1)
    
    #Determine threshold    
    d["threshold"] = leaf.parent.split
    
    # Keep track of the leaf id and the leaf prediction
    d["leaf_id"] = leaf_id
    d["prediction"] = leaf_weight
    
    df_new = DataFrame(d)
    
    append!(df, df_new)
    update_leaf_df!(df, leaf.parent, cols, leaf_id, leaf_weight)
end

function find_leaf_df(tree::AbstractJLBoostTree, bbl::BlackBoxLearner)

    df = DataFrame()
    leaves = find_all_leaves(tree);
    #leaf = leaves[1];
    cols = names(bbl.X);

    i = 1
    for leaf in leaves 
        update_leaf_df!(df, leaf, cols, i, leaf.weight)
        i += 1
    end
    
    return df
end

function embed_single_tree(gm::GlobalModel, bbl::BlackBoxLearner, tree_id::Int64, tree::AbstractJLBoostTree; M=1000)
    m = gm.model;
    cols = names(bbl.X)
    x = bbl.vars;
    
    # Calculate the df that describes the constraints
    df = find_leaf_df(tree, bbl)
    
    # Predictions for each leaf
    leaf_predictions = combine(first, groupby(df, :leaf_id))[!, "prediction"];

    # Splitting thresholds
    intercept = df[!, "threshold"];

    # Splitting coefficients (i.e. which variable is active)
    coeffs = Matrix(df[!, cols]);

    # Leaf ids
    l_ids = convert.(Int64, df[!, "leaf_id"]);
    n_leaves = length(unique(l_ids))

    # Create 1 variable for every leaf
    var_name = eval(Meta.parse(":d$(tree_id)"));
    m[var_name] = @variable(m, [i=1:n_leaves], Bin, base_name=string(var_name));
    leaf_vars = m[var_name];

    # Create 1 outcome variable for the whole tree
    var_name = eval(Meta.parse(":y$(tree_id)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    outcome_var = m[var_name]

    @constraint(m, coeffs*x .<= intercept.+M*(1 .- leaf_vars[l_ids]));
    @constraint(m, outcome_var == leaf_predictions'*leaf_vars);
    @constraint(m, sum(leaf_vars[i] for i=1:n_leaves) == 1);
    
    return outcome_var;
end

function gbm_embed_helper(lnr::Union{GBM_Regressor, GBM_Classifier}, gm::GlobalModel, bbl::BlackBoxClassifier, lb=-Inf, ub=Inf; kwargs...) 
    
    trees = lnr.gbm.jlt;
    
    m = gm.model;
    
    outcome_vars = []
    etas = []
    for (i, tree) in enumerate(trees)
        tree_outcome = embed_single_tree(gm, bbl, i, tree; M=1000);
        push!(outcome_vars, tree_outcome)
        push!(etas, tree.eta)
    end

    # Define final outcome variable
    var_name = eval(Meta.parse(":$(bbl.name)"));
    m[var_name] = @variable(m, base_name=string(var_name));
    final_outcome = m[var_name];
    
    if lb != -Inf
        @constraint(m, final_outcome >= lb);
    end
    if ub != Inf
        @constraint(m, final_outcome <= ub);
    end
    
    # Final outcome variable is the weighted average
    # of the sub-trees
    @constraint(m, outcome_vars'*etas./sum(etas) == final_outcome)
    
    return Dict(), Dict()
end

"""
Embed MIO constraints on GBM classifier
"""
function gbm_cl_embed(lnr::GBM_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)

   return gbm_embed_helper(lnr, gm, bbl, 0.5)
end

"""
Embed MIO constraints on GBM regressor
"""
function gbm_cl_embed(lnr::GBM_Classifier, gm::GlobalModel, bbl::BlackBoxClassifier; kwargs...)

   return gbm_embed_helper(lnr, gm, bbl, 0.5)
end


"""
Test GBM 
"""

# function test_gbm()

#     X = rand(100, 50)
    
#     y = 1.0*(sum(X', dims=[1]).>50)[1,:]#

#     lnr = GBM_Classifier()

#     lnr.fit!(lnr, X, y);
    
#     y_hat = lnr.predict(lnr, X)
#     accuracy = 100*sum(y .== y_hat)/length(y)
#     println("Accuracy $(accuracy)")
    
#     y_hat = lnr.predict(lnr, X; continuous=true)
#     accuracy = 100*sum(y .== y_hat)/length(y)
#     println("Accuracy $(accuracy)")
    
#     return y, y_hat
# end
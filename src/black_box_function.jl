#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

@with_kw mutable struct BlackBoxFunction
"""
Contains all required info to be able to generate a global optimization constraint.
"""
    name::Union{String, Real} = ""                     # Function name
    constraint::Union{JuMP.ConstraintRef, JuMP.NonlinearExpression} # The JuMP constraint function
    vars::Array{JuMP.VariableRef}                      # JuMP variables
    X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars))  # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    n_samples::Int = 100                               # For next set of samples, set and forget.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbf.constraint, x)
end

function eval!(bbf::BlackBoxFunction, X::DataFrame)
    """ Evaluates BlackBoxFunction at all X and stores the resulting data. """
    values = bbf(X);
    append!(bbf.X, X[:,string.(bbf.vars)], cols=:setequal)
    append!(bbf.Y, values);
    bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    return
end



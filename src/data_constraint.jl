#=
data_constraint:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-12
=#


"""
    @with_kw mutable struct DataConstraint

Contains all required info to be able to generate a global optimization constraint from data only.
Mandatory arguments:
    vars::Array{JuMP.VariableRef,1}

Can be tagged with additional info.
"""

@with_kw mutable struct DataConstraint
    vars::Array{JuMP.VariableRef,1}                      # JuMP variables (flat)
    name::Union{String, Real} = ""                     # Function name
    X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars))
                                                       # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    tags::Array{String} = []                           # Other tags
end
#=
data_constraint:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-12
=#

""" Returns default DataConstraint settings for approximation."""
function dc_defaults()
    settings = Dict(:threshold_accuracy => 0.95,      # Minimum tree accuracy
                    :threshold_feasibility => 0.15,   # Minimum feasibility ratio
                    :reloaded => false)
end


"""
    @with_kw mutable struct DataConstraint

Contains all required info to be able to generate a global optimization constraint from data only.
Mandatory arguments:
    vars::Array{JuMP.VariableRef,1}

Can be tagged with additional info.
"""

@with_kw mutable struct DataConstraint
    xvars::Array{JuMP.VariableRef,1}                      # Input JuMP variables (flat)
    yvars::Union{Array{JuMP.VariableRef, 1}, Nothing}     # Output JuMP variables (flat)
    name::String = ""                                     # Function name
    X::DataFrame = DataFrame([Float64 for i=1:length(xvars)], string.(xvars))
                                                       # Function inputs
    Y::DataFrame = DataFrame()                         # Function outputs
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    Y_regressions:: Union{Dict, Nothing} = nothing    # Regressions over leaves
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    settings = dc_defaults()                           # Relevant settings
end

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
    xvars::Array{JuMP.VariableRef,1}                      # Input JuMP variables (flat)
    yvars::Union{Array{JuMP.VariableRef, 1}, Nothing}     # Output JuMP variables (flat)
    name::Union{String, Real} = ""                     # Function name
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
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    tags::Array{String} = []                           # Other tags
end

""" Adds data to a DataConstraint. """
function add_data!(dc::DataConstraint, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    append!(dc.X, X[:,string.(dc.vars)], cols=:setequal)
    append!(dc.Y, Y)
    dc.feas_ratio = sum(dc.Y .>= 0)/length(dc.Y); #TODO: optimize.
    return
end

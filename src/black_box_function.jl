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
    fn::Function                                       # The function
    vks::Array                                         # Varkeys
    lbs::Dict = Dict(vks .=> -Inf)                     # Variable lower bounds
    ubs::Dict = Dict(vks .=> Inf)                      # Variable upper bounds
    X::DataFrame = DataFrame([Float64 for i in vks], vks)  # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    constraints::Array = []                            # and their corresponding constraints,
    accuracies::Array{Float64} = []                    # and their scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    n_samples::Int = 100                               # For next set of samples, set and forget.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})
    if isa(x, Union{DataFrameRow, Dict})
        return bbf.fn(x);
    elseif isa(x, DataFrame)
        return [bbf.fn(x[i,:]) for i=1:size(x,1)]
    else
        throw(OCTException("This datatype is not supported for BlackBoxFunction evaluation."))
    end
end

function eval!(bbf::BlackBoxFunction, X::DataFrame)
    """ Evaluates BlackBoxFunction at all X and stores the resulting data. """
    values = bbf(X);
    append!(bbf.X, X[:,bbf.vks], cols=:setequal)
    append!(bbf.Y, values);
    bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    return
end


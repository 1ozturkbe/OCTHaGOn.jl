#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

using DataFrames
using Distributions
using GaussianProcesses
using NearestNeighbors
using Parameters

include("exceptions.jl")
include("learners.jl")

@with_kw mutable struct BlackBoxFunction
"""
Contains all required info to be able to generate a global optimization constraint.
"""
    name::Union{String, Int} = ""                      # Function name
    fn::Function                                       # The function
    vks::Array                                         # Varkeys
    lbs::Dict = Dict(vks .=> -Inf)                     # Variable lower bounds
    ubs::Dict = Dict(vks .=> Inf)                      # Variable upper bounds
    X::DataFrame = DataFrame([Float64 for i in vks], vks)  # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    gp::Union{GPE, Nothing} = nothing                  # Gaussian Process
    learners::Array{IAI.GridSearch} = []               # Learners...
    constraints::Dict = Dict()                         # and their corresponding constraints,
    accuracies::Array{Float64} = []                    # and their scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    n_samples::Int = 100                               # For next set of samples.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})
    if isa(x, Union{DataFrameRow, Dict})
        return bbf.fn(x);
    elseif isa(x, DataFrame)
        return [bbf.fn(x[i,:]) for i=1:size(x,1)]
    else
        throw(OCT.OCTException("This datatype is not supported for BlackBoxFunction evaluation."))
    end
end

function eval!(bbf::BlackBoxFunction, X::DataFrame)
    """ Evaluates BlackBoxFunction at all X and stores the resulting data. """
    if isnothing(bbf.X)
        bbf.X = X[:, bbf.vks];
        bbf.Y = bbf(X);
        bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y);
    else
        values = bbf(X);
        append!(bbf.X, X[:,bbf.vks], cols=:setequal)
        append!(bbf.Y, values);
        bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    end
    return
end

function predict(bbf::BlackBoxFunction, X::AbstractArray)
    μ, σ = predict_f(bbf.gp, transpose(X))
    return μ, σ
end

function optimize_gp!(bbf::BlackBoxFunction)
    """ Optimizes a GaussianProcess over a BlackBoxFunction,
    with adaptively changing kernel. """
#         bbf.gp = ElasticGPE(length(bbf.idxs), # data
#         mean = MeanConst(sum(bbf.Y)/length(bbf.Y)), logNoise = -10)
    lbs = [bbf.lbs[key] for key in bbf.vks];
    ubs = [bbf.ubs[key] for key in bbf.vks];
    bbf.gp = GPE(transpose(Array(bbf.X)), bbf.Y, # data
    MeanConst(sum(bbf.Y)/length(bbf.Y)),
    SEArd(log.((ubs-lbs)./(2*sqrt(length(bbf.Y)))), -5.))
    optimize!(bbf.gp); #TODO: optimize GP
                       # Instead of regenerating at every run, figure out
                       # how to update.
end


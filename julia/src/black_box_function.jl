#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

using DataFrames
using Distributions
using Parameters
using GaussianProcesses
using Plots

include("exceptions.jl")
include("learners.jl")

@with_kw mutable struct BlackBoxFn
"""
Contains all required info to be able to generate a global optimization problem.
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
    constraints::Array{Array} = Array[]                # and their corresponding constraints,
    accuracies::Array{Float64} = []                    # and their scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    n_samples::Int = 100                               # For next set of samples.
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFn)(x::Union{DataFrame,Dict,DataFrameRow})
    if isa(x, Union{DataFrameRow, Dict})
        return bbf.fn(x);
    elseif isa(x, DataFrame)
        return [bbf.fn(x[i,:]) for i=1:size(x,1)]
    else
        throw(OCT.OCTException("This datatype is not supported for BlackBoxFn evaluation."))
    end
end

function eval!(bbf::BlackBoxFn, X::DataFrame)
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

function predict(bbf::BlackBoxFn, X::AbstractArray)
    μ, σ = predict_f(bbf.gp, transpose(X))
    return μ, σ
end

function optimize_gp!(bbf::BlackBoxFn)
    """ Optimizes a GaussianProcess over a BlackBoxFn,
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


function sample_and_eval!(bbf::BlackBoxFn;
                          ratio=10.)
    """ Samples and evaluates BlackBoxFn, with n_samples new samples.
    ratio*n_samples is how many random LHC samples are generated for prediction from GP. """
    vks = bbf.vks;
    n_dims = length(vks);
    if isnothing(bbf.gp) # If we don't have any GPs yet, uniform sample.
       plan, _ = LHCoptim(bbf.n_samples, n_dims, 3);
       if any(isinf.(values(bbf.lbs))) || any(isinf.(values(bbf.ubs)))
           throw(OCTException("Model is not properly bounded."))
       end
       df = DataFrame(scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i in vks]), vks);
       eval!(bbf, df);
    else
        plan = randomLHC(Int(round(bbf.n_samples*ratio)), n_dims);
        random_samples = scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i in vks]);
        μ, σ = predict_f(bbf.gp, random_samples');
        cdf_0 = [Distributions.cdf(Distributions.Normal(μ[i], σ[i]),0) for i=1:size(random_samples,1)];
         #TODO: add criterion for information as well (something like sortperm(σ))
        # Sample places with high probability of being near boundary (0),
        # but also balance feasibility ratio.
        p = bbf.feas_ratio
#         balance_fn = x -> -1*tan(2*atan(-0.5)*(x-0.5)) + 0.5
        balance_fn = x -> -1/0.5^2*(x-0.5)^3 + 0.5;
        indices = sortperm(abs.(cdf_0 .- balance_fn(p)));
        samples = DataFrame(random_samples[indices[1:bbf.n_samples],:], vks);
        eval!(bbf, samples);
    end
    return
end
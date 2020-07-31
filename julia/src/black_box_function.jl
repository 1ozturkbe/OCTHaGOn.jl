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
    idxs::Union{Array, Nothing} = nothing              # Variable indices
    lbs::Array = []                                    # Variable lower bounds
    ubs::Array = []                                    # Variable upper bounds
    samples::Union{Nothing, DataFrame} = nothing       # Function samples
    values::Array = []                                 # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    feas_min::Float64 = 0.1                            # Minimum feas_ratio
    equality::Bool = false                             # Equality check
    gp::Union{GPE, Nothing} = nothing                  # Gaussian Process
    learners::Array{IAI.GridSearch} = []               # Learners...
    constraints::Array{Array} = Array[]                # and their corresponding constraints,
    accuracies::Array{Float64} = []                    # and their scores.
    n_samples::Int = 100                               # For next set of samples.
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFn)(x)
    return bbf.fn(x)
end

function eval!(bbf::BlackBoxFn, X::Union{AbstractArray, DataFrame})
    df = DataFrame(X);
    if isnothing(bbf.samples)
        bbf.samples = df;
        bbf.values = [bbf(df[i,:]) for i=1:size(df,1)];
        bbf.feas_ratio = sum(bbf.values .>= 0)/length(bbf.values);
    else
        values = [bbf(df[i,:]) for i=1:size(df,1)];
        append!(bbf.samples, df, cols=:setequal);
        append!(bbf.values, values);
        bbf.feas_ratio = sum(bbf.values .>= 0)/length(bbf.values); #TODO: optimize.
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
#         mean = MeanConst(sum(bbf.values)/length(bbf.values)), logNoise = -10)
    bbf.gp = GPE(transpose(Array(bbf.samples)), bbf.values, # data
    MeanConst(sum(bbf.values)/length(bbf.values)),
    SEArd(log.((bbf.ubs - bbf.lbs)./(2*length(bbf.values))), -5.))
    optimize!(bbf.gp); #TODO: optimize GP
                       # Instead of regenerating at every run, figure out
                       # how to update.
end


function sample_and_eval!(bbf::BlackBoxFn; ratio=10.)
    """ Samples and evaluates BlackBoxFn, with n_samples new samples.
    ratio*n_samples is how many random LHC samples are generated for prediction from GP. """
    n_dims = length(bbf.idxs);
    if isnothing(bbf.gp) # If we don't have any GPs yet, uniform sample.
       plan, _ = LHCoptim(bbf.n_samples, n_dims, 3);
       if any(isinf.(hcat(bbf.lbs, bbf.ubs)))
           throw(ArgumentError("Model is not properly bounded."))
       end
       df = DataFrame(scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i=1:n_dims]));
       eval!(bbf, df);
    else
        plan = randomLHC(Int(round(bbf.n_samples*ratio)), n_dims);
        random_samples = scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i=1:n_dims]);
        μ, σ = predict_y(bbf.gp, random_samples');
        cdf_0 = [Distributions.cdf(Distributions.Normal(μ[i], σ[i]),0) for i=1:size(random_samples,1)];
         #TODO: add criterion for information as well (something like sortperm(σ))
        # Sample places with high probability of being near boundary (0),
        # but also balance feasibility ratio.
        p = bbf.feas_ratio
        balance_point = -1/0.5^2*(p-0.5)^3 + 0.5;
        indices = sortperm(abs.(cdf_0 .- balance_point));
        eval!(bbf, random_samples[indices[1:bbf.n_samples],:])
    end
    return
end
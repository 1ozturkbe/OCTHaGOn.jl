#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

using DataFrames
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
    equality::Bool = false                             # Equality check
    gp::Union{GPE, Nothing} = nothing                  # Gaussian Process
    learners::Array{IAI.GridSearch} = []               # Learners...
    constraints::Array{Array} = Array[]                # and their corresponding constraints
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
    else
        append!(bbf.samples, df, cols=:setequal);
        append!(bbf.values, [bbf(df[i,:]) for i=1:size(df,1)]);
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
    optimize!(bbf.gp);
end


function sample_and_eval!(bbf::BlackBoxFn; n_samples=1000, ratio=10.)
    """ Samples and evaluates BlackBoxFn, with n_samples new samples.
    ratio*n_samples is how many random LHC samples are generated for prediction from GP. """
    n_dims = length(bbf.idxs);
    if isnothing(bbf.gp) # If we don't have any GPs yet, uniform sample.
       plan, _ = LHCoptim(n_samples, n_dims, 3);
       if any(isinf.(hcat(bbf.lbs, bbf.ubs)))
           throw(ArgumentError("Model is not properly bounded."))
       end
       df = DataFrame(scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i=1:n_dims]));
       eval!(bbf, df);
    else
        plan = randomLHC(Int(round(n_samples*ratio)), n_dims);
        random_samples = scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i=1:n_dims]);
        μ, σ = predict_y(bbf.gp, random_samples');
        indices = sortperm(abs.(μ)) # Sample places with high probability of being near boundary (0)
        eval!(bbf, random_samples[indices[1:n_samples],:])
    end
    return
end
#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

using DataFrames
using Parameters
using GaussianProcesses

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

function eval!(bbf::BlackBoxFn, X::AbstractArray)
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
    μ, σ = predict_f(bbf.gp, X)
    return μ, σ
end

function optimize_gp!(bbf::BlackBoxFn)
    if isnothing(bbf.gp)
        bbf.gp = ElasticGPE(Matrix(bbf.samples), bbf.values, # data
        MeanConst(sum(bbf.values)/length(bbf.values)),       # constant mean
        SEArd(log.((bbf.ubs - bbf.lbs)./20.), -5.))          # SE kernel
    end
    optimize!(bbf.gp)
end


# function sample_and_eval!(bbf, md::ModelData; n_samples=1000)
#     if isnothing(idxs)
#         bbf.idxs = [i for i=1:length(md.c)];
#     end
#     n_dims = length(bbf.idxs);
#     if isnothing(bbf.gp)
#        plan, _ = LHCoptim(n_samples, n_dims, 3);
#        if any(isinf.(hcat(md.lbs, md.ubs)))
#            throw(ArgumentError("Model is not properly bounded."))
#        end
#        df = DataFrame(scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]));
#        eval!(bbf, df);
#     end
#     return
# end
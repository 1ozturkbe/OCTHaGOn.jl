#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

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
    samples::Array{Array} = Array[]                    # Inequality function samples
    equality::Bool = false                             # Equality check
    gp::Union{Nothing} = nothing                       # Gaussian Process
    learners::Array{IAI.GridSearch} = [gridify(base_otc())]     # Learners...
    constraints::Array{Array} = Array[]                # and their corresponding constraints
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFn)(x)
    return bbf.fn(x)
end

function optimize_gp!(bbf::BlackBoxFn)
    optimize!(bbf.gp)
end

# function sample_and_eval!(bbf, md::ModelData; n_samples=1000)
#     if isnothing(idxs)
#         bbf.idxs = [i for i=1:length(md.c)];
#     end
#     n_dims = length(bbf.idxs);
#     if isnothing(bbf.gp):
#         plan, _ = LHCoptim(n_samples, n_dims, 3);
#        if any(isinf.(hcat(md.lbs, md.ubs)))
#            throw(ArgumentError("Model is not properly bounded."))
#        end
#        X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
#
#     return
# end
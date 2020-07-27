#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

using Parameters

include("exceptions.jl");

@with_kw mutable struct BlackBoxFn
"""
Contains all required info to be able to generate a global optimization problem.
"""
    name::Union{String, Int} = ""                      # Function name
    fn::Function                                       # The function
    idxs::Union{Nothing, Array} = Nothing              # Variable indices
    samples::Array{Array} = Array[]                    # Inequality function samples
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    constraints::Array{Array} = Array[]                # and their corresponding constraints
    tags::Array{String} = []                           # Other tags
end

function (bbf::BlackBoxFn)(x)
    return bbf.fn(x)
end
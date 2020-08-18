module OptimalConstraintTree
    using Combinatorics
    using ConicBenchmarkUtilities
    using DataFrames
    using Distributions
    using GaussianProcesses
    using Gurobi
    using JuMP
    using LatinHypercubeSampling
    using Main.IAI
    using MathOptInterface
    using NearestNeighbors
    using Parameters
    using Plots
    using PyCall
    using Random
    using SparseArrays

    include("learners.jl")

    include("exceptions.jl")

    include("black_box_function.jl")

    include("root_finding.jl")

    include("post_process.jl")

    include("bin_to_leaves.jl")

    include("augment.jl")

    include("model_data.jl")

    include("constraintify.jl")
    include("fit.jl")

    include("convexRegress.jl")

    include("plot.jl")

    include("tools.jl")

           # Structs
    export ModelData, BlackBoxFunction,
           # Functions on ModelData and BlackBoxFunctions
           gridify, learn_from_data!, find_bounds!, update_bounds!
           lh_sample, boundary_sample, add_bounds!,
           accuracy, feasibility, accuracy_check, feasibility_check, solve,
           # Functions on BlackBoxFunctions
           eval!, sample_and_eval!, plot, learn_constraint!,
           add_fn!, add_linear_ineq!, add_linear_eq!,
           secant_method, knn_sample,
           # Functions on JuMP.Models
           add_feas_constraints!, add_regr_constraints!,
           add_linear_constraints!, add_tree_constraints!,
           base_otr, base_otc, update_bounds!, sample, jump_it!,
           sagemark_to_ModelData,
           # Exceptions
           OCTException
end


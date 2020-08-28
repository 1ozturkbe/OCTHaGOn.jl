module OptimalConstraintTree
    using Combinatorics
    using ConicBenchmarkUtilities
    using DataFrames
    using Distributions
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
           # ModelData Functions
           fns_by_feasibility,
           # Functions on ModelData and BlackBoxFunctions
           gridify, learn_from_data!, find_bounds!, update_bounds!,
           lh_sample, boundary_sample, add_bounds!,
           accuracy, feasibility, accuracy_check, feasibility_check, globalsolve,
           get_solution, evaluate_feasibility,
           # Functions on BlackBoxFunctions
           eval!, sample_and_eval!, plot, learn_constraint!,
           add_fn!, add_linear_ineq!, add_linear_eq!,
           secant_method, knn_sample, build_knn_tree,
           find_knn, classify_patches,
           # Functions on JuMP.Models
           add_feas_constraints!, add_regr_constraints!,
           add_linear_constraints!, add_tree_constraints!,
           base_otr, base_otc, update_bounds!, sample, jump_it!,
           # Functions to import global optimization problems,
           sagemark_to_ModelData, CBF_to_MOI, CBF_to_ModelData,
           # Exceptions
           OCTException,
           # Display and plotting
           show_trees, plot_2d, plot_2d_predictions, plot_accuracies
end


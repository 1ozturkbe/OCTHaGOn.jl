module OptimalConstraintTree
    using Combinatorics
    using ConicBenchmarkUtilities
    using DataFrames
    using Gurobi
    using JuMP
    using LatinHypercubeSampling
    using Main.IAI
    using MathOptInterface
    using MathOptSetDistances
    using NearestNeighbors
    using Parameters
#     using Plots
    using ProgressMeter
    using PyCall
    using Random

    include("small_scripts.jl")

    include("on_jump.jl")

    include("learners.jl")

    include("exceptions.jl")

    include("black_box_function.jl")

    include("root_finding.jl")

    include("post_process.jl")

    include("iai_wrappers.jl")

    include("augment.jl")

    include("global_model.jl")

    include("constraintify.jl")

    include("fit.jl")

    include("plot.jl")

    include("tools.jl")

           # Structs
    export GlobalModel, BlackBoxFunction,
           # GlobalModel Functions
           fns_by_feasibility, globalsolve,
           # JuMP.Model extensions to GlobalModel
#            set_optimizer, optimize!, JuMP.all_variables,
           # Functions on GlobalModel and BlackBoxFunctions
           gridify, learn_from_data!, find_bounds!, add_nonlinear_constraint,
           lh_sample, boundary_sample,
           accuracy, feasibility, check_accuracy, check_feasibility, check_bounds,
           solution, evaluate_feasibility,
           nonlinearize!,
           # Functions on BlackBoxFunctions
           eval!, sample_and_eval!, plot, learn_constraint!,
           secant_method, knn_sample, build_knn_tree,
           find_knn, classify_patches,
           # Functions on JuMP objects
           evaluate, fetch_variable, get_bounds,
           linearize_objective!, classify_constraints,
           bound!,
           data_to_DataFrame, data_to_Dict,
           distance_to_set, get_constant,
           add_feas_constraints!, add_regr_constraints!,
           add_tree_constraints!, clear_tree_constraints!,
           base_otr, base_otc, sample,
           functionify,
           # Functions to import global optimization problems,
           sagemark_to_GlobalModel,
           alphac_to_expr, alphac_to_objexpr, alphac_to_varbound!,
           # Exceptions
           OCTException,
           # Display and plotting
           show_trees,
#            plot_2d, plot_2d_predictions, plot_accuracies
           # Debugging tools
            clear_data!, chop_dict,
            # Small scripts
            vars_from_expr, get_varmap, deconstruct, flat,
            infarray
end


module OptimalConstraintTree
    using Combinatorics
    using ConicBenchmarkUtilities
    using DataFrames
    using Gurobi
    using JuMP
    using LatinHypercubeSampling
    using MacroTools
    using Main.IAI
    using MathOptInterface
    using MathOptSetDistances
    using NearestNeighbors
    using Parameters
#     using Plots
    using ProgressMeter
    using Random

    include("small_scripts.jl")

    include("on_jump.jl")

    include("learners.jl")

    include("exceptions.jl")

    include("black_box_function.jl")

    include("data_constraint.jl")

    include("root_finding.jl")

    include("post_process.jl")

    include("iai_wrappers.jl")

    include("augment.jl")

    include("global_model.jl")

    include("constraintify.jl")

    include("fit.jl")

    include("plot.jl")

           # Structs
    export GlobalModel, BlackBoxFunction, DataConstraint,
           # GlobalModel Functions
           fns_by_feasibility, globalsolve,
           # JuMP.Model extensions to GlobalModel
           set_optimizer, optimize!,
           # Functions on GlobalModel and BlackBoxFunctions and DataConstraints
           gridify, get_learner, learn_from_data!, find_bounds!, add_nonlinear_constraint,
           add_nonlinear_or_compatible,
           lh_sample, boundary_sample,
           accuracy, feasibility, check_accuracy, check_feasibility, check_bounds,
           solution, evaluate_feasibility,
           nonlinearize!,
           # Functions on both BlackBoxFunctions and DataConstraints
           show_trees, learn_constraint!,
           # Functions on BlackBoxFunctions only
           eval!, sample_and_eval!,
           secant_method, knn_sample, build_knn_tree,
           find_knn, classify_patches,
           # Functions on DataConstraints only
           add_data!,
           # Functions on IAI structs
           bin_to_leaves, regress,
           # Functions on JuMP objects
           variable_by_symbol,
           evaluate, fetch_variable, get_bounds,
           linearize_objective!, classify_constraints,
           bound!,
           data_to_DataFrame, data_to_Dict,
           distance_to_set, get_constant,
           add_feas_constraints!, add_regr_constraints!,
           add_tree_constraints!, clear_tree_constraints!,
           base_otr, base_otc,
           functionify,
           # Functions to import global optimization problems,
           sagemark_to_GlobalModel,
           alphac_to_expr, alphac_to_objexpr, alphac_to_varbound!,
           # Exceptions
           OCTException,
           # Display and plotting
#            plot_2d, plot_2d_predictions, plot_accuracies
           # Debugging tools
            clear_data!, chop_dict,
            # Small scripts
            vars_from_expr, get_varmap, deconstruct, flat,
            infarray, substitute, substitute_expr, substitute_args
end


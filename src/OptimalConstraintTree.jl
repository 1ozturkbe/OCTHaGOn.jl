module OptimalConstraintTree
    using Combinatorics
    using CPLEX
    using DataFrames
    using ForwardDiff
    using JLD
    using JuMP
    using LatinHypercubeSampling
    using LinearAlgebra
    using MacroTools
    using Main.IAI
    using MathOptInterface
    using MathOptSetDistances
    using Missings
    using NearestNeighbors
    using Parameters
    using ProgressMeter
    using Random
    using StatsBase

    const PROJECT_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = PROJECT_ROOT * "\\data\\"
    const BARON_DIR = PROJECT_ROOT * "\\data\\baron\\"
    const GAMS_DIR = PROJECT_ROOT * "\\data\\gams\\"  
    const SOL_DIR = PROJECT_ROOT * "\\data\\solutions\\"
    const TREE_DIR = PROJECT_ROOT * "\\data\\trees\\"
    const CPLEX_SILENT = with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0)

    include("small_scripts.jl")

    include("on_jump.jl")

    include("learners.jl")

    include("exceptions.jl")

    include("settings.jl")
    
    include("black_box_learners.jl")

    include("root_finding.jl")

    include("iai_wrappers.jl")

    include("global_model.jl")

    include("sample.jl")

    include("constraintify.jl")

    include("fit.jl")

    include("refine.jl")

    include("optimize.jl")

    include("plot.jl")
        
        # Structs
    export GlobalModel, BlackBoxLearner, BlackBoxClassifier, BlackBoxRegressor,
        # GlobalModel Functions
        surveysolve,
        # JuMP.Model extensions to GlobalModel
        set_optimizer, optimize!,
        # Functions on GlobalModel and BlackBoxLearners
        learn_from_data!, find_bounds!, find_linear_bounds!,
        add_nonlinear_constraint, add_nonlinear_or_compatible, determine_vars,
        lh_sample, boundary_sample, last_leaf_sample,
        feasibility_sample, 
        feasibility, check_accuracy, check_feasibility, 
        check_bounds,  check_sampled, update_local_convexity, update_vexity,
        solution, 
        evaluate_feasibility, evaluate_accuracy,
        feas_gap, 
        nonlinearize!,
        save_fit, load_fit, set_param, get_param, 
        # Functions on BlackBoxLearners
        show_trees, learn_constraint!, 
        all_mi_constraints, active_lower_tree,
        find_leaf_of_soln, evaluate, evaluate_gradient, update_gradients,
        # Functions on BlackBoxLearners only
        eval!, uniform_sample_and_eval!,
        secant_method, knn_sample, build_knn_tree,
        find_knn, classify_patches, classify_curvature,
        # Functions on BlackBoxLearners
        add_data!, match_bbls_to_vars,
        # Functions on IAI structs
        find_leaves, regress, check_if_trained, trust_region_data, pwl_constraint_data,
        # Functions on JuMP objects
        fetch_variable, get_bounds, get_unbounds,
        linearize_objective!, classify_constraints,
        bound!,
        data_to_DataFrame, data_to_Dict,
        distance_to_set, get_constant, 
        add_feas_constraints!, add_regr_constraints!,
        add_tree_constraints!, clear_tree_constraints!, update_tree_constraints!,
        base_regressor, base_classifier,
        fit_regressor_kwargs, fit_classifier_kwargs, 
        regressor_kwargs, classifier_kwargs,
        functionify, gradientify, 
        # Other approximators, 
        ridge_regress, u_regress, l_regress, boundify, svm, reweight,
        add_infeasibility_cuts!,
        # Functions to import global optimization problems,
        sagemark_to_GlobalModel,
        alphac_to_expr, alphac_to_objexpr, alphac_to_varbound!,
        # Exceptions
        OCTException,
        # Debugging tools
        clear_data!, clear_tree_data!,
        # Small scripts
        vars_from_expr, vars_from_constraint, get_varmap, get_var_ranges, get_datamap,
        deconstruct, flat,
        zeroarray, substitute, power, normalized_data, truncate_sigfigs
end


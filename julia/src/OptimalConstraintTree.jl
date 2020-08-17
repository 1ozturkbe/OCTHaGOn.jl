module OptimalConstraintTree

    using Main.IAI

    include("augment.jl")
    include("black_box_function.jl")
    include("bin_to_leaves.jl")
    include("constraintify.jl")
    include("convexRegress.jl")
    include("fit.jl")
    include("learners.jl")
    include("model_data.jl")
    include("plot.jl")
    include("root_finding.jl")
    include("tools.jl")

           # Structs
    export ModelData, BlackBoxFunction,
           # Functions on ModelData and BlackBoxFunctions
           gridify, learn_from_data!, find_bounds!, update_bounds!
           lh_sample, boundary_sample, add_bounds!,
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


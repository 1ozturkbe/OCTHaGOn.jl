module OptimalConstraintTree

    using Main.IAI

    include("augment.jl")
    include("black_box_function.jl")
    include("bin_to_leaves.jl")
    include("constraintify.jl")
    include("convexRegress.jl")
    include("fit.jl")
    include("model_data.jl")
    include("tools.jl")

    export ModelData, BlackBoxFn, gridify, learn_constraints!, learn_from_data!, find_bounds!
           add_feas_constraints!, add_regr_constraints!,
           add_fn!, add_linear_ineq!, add_linear_eq!,
           add_linear_constraints!, add_tree_constraints!,
           base_otr, base_otc, update_bounds!, sample, jump_it!,
           show_trees, sagemark_to_ModelData
end


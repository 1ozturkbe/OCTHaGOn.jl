module OptimalConstraintTree

    using Main.IAI

    include("augment.jl")
    include("bin_to_leaves.jl")
    include("constraintify.jl")
    include("convexRegress.jl")
    include("doe.jl")
    include("fit.jl")
    include("model_data.jl")
    include("tools.jl")

    export ModelData, learn_constraints, find_bounds!
           add_feas_constraints!, add_regr_constraints!,
           add_linear_constraints!, add_tree_constraints!,
           base_otr, base_otc, update_bounds!, sample, jump_it,
           show_trees, sagemark_to_ModelData
end


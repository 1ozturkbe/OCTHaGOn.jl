module OptimalConstraintTree

include("augment.jl")
include("bin_to_leaves.jl")
include("constraintify.jl")
include("convexRegress.jl")
include("doe.jl")
include("fit.jl")
include("model_data.jl")
include("tools.jl")

export ModelData, learn_constraints!, learn_objective!,
       add_feas_constraints!, add_mio_constraints!,
       base_otr, base_otc, update_bounds!, sample, jumpit
end
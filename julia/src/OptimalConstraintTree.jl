module OptimalConstraintTree

include("augment.jl")
include("bin_to_leaves.jl")
include("convexRegress.jl")
include("doe.jl")
include("constraintify.jl")
include("fit.jl")
include("structs.jl")

export ModelData, learn_constraints!, learn_objective!,
       add_feas_constraints!, add_mio_constraints!, import_sagebenchmark,
       base_otr, base_otc, update_bounds!
end

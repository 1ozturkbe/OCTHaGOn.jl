module OptimalConstraintTree

include("augment.jl")
include("bin_to_leaves.jl")
include("convexRegress.jl")
include("doe.jl")
include("constraintify.jl")
include("regress.jl")
include("structs.jl")
include("../test/examples.jl")

export function_model, learn_constraints, learn_objective,
       add_feas_constraints!, add_mio_constraints!, import_sagebenchmark,
       base_otr, base_otc

end

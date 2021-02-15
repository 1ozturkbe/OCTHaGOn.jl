include("../load.jl")

using Distributed
addprocs(4)

gms = [minlp(true), nlp1(true), nlp2(true), nlp3(true), gear(true), speed_reducer()]

uniform_sample_and_eval!.(gms)


# gm = minlp(true)

function test_curvature(gm::GlobalModel)
    uniform_sample_and_eval!(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    for bbc in bbcs
        learn_constraint!(bbc)
    end
    try
        bbr = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor][1]
        classify_curvature(bbr)
    catch
        println("GM $(gm.name) does not have a Regressor. ")
    end
    return gm
end

gm = speed_reducer()
set_param(gm, :ignore_feasibility, true)
uniform_sample_and_eval!(gm)
bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier];
learn_constraint!.(bbcs);
[add_tree_constraints!(gm, bbc) for bbc in bbcs];
bbr = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor][1]
classify_curvature(bbr)
classify_curvature.(bbcs)
convex_prop = [sum(bbc.curvatures .>= 0)/size(bbc.X, 1) for bbc in bbcs]
learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6)
update_tree_constraints!(gm, bbr)
optimize!(gm)

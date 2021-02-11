include("../load.jl")

using Distributed
addprocs(4)

gms = [minlp(true), nlp1(true), nlp2(true), nlp3(true), gear(true), speed_reducer()]

uniform_sample_and_eval!.(gms)

learn_constraint!.(gms)

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


# function update_vexity(bbr::BlackBoxRegressor)
#     idx = active_lower_tree(bbr)
#     lnr = bbr.learners[idx]
#     leaves = find_leaves(bbr.learners[end])


#     bbr.vexity = Dict(key => nothing for (key, value) in bbr.ul_data if key <= 0)





function feasibility_sample(bbc::BlackBoxClassifier)
    lnr = learn_from_data!(bbc.X, bbc.Y .>= 0, base_classifier())
    all_leaves = find_leaves(lnr)
    feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
        # Equal sampling in each leaf. 

        
function feasibility_sample(gm::GlobalModel)
    for bbl in gm.bbls
        if bbl isa BlackBoxClassifier && bbc.feas_ratio <= get_param(bbc, :threshold_feasibility)
            feasibility_sample(bbl)
        end
    end
    return
end


# if sum(bbr.curvatures .> 0 >= 0.5*size(bbr.X, 1)) # if some convex properties...
#     if sum(bbr.curvatures .> 0.98 * size(bbr.X, 1))
#         idxs = Int64.(round.(rand(10) .* size(bbr.X, 1)))
#         thresh = (maximum(bbr.Y) - minimum(bbr.Y)) * 1e-10
#         curvs = []
#         diffs = [[Array(bbr.X[i, :]) - Array(bbr.X[j, :]) for j in idxs] for i in idxs]
#         grads = bbr.gradients[idxs, :]
#         under_offsets = [-dot(grads[i], diffs[i][j])]
#             under_offsets = [-dot(center_grad, differ) for differ in diffs]
#             actual_offsets = bbl.Y[knn_idxs[i]] .- bbl.Y[i]
#             if all(actual_offsets - under_offsets .>= -thresh)
#                 bbl.curvatures[i] = 1
#             elseif all(actual_offsets - under_offsets .<= thresh)
#                 bbl.curvatures[i] = -1
#             else
#                 bbl.curvatures[i] = 0
#             end
#         end
# else
# end
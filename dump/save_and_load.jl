"""
    save_fit(bbl::Union{GlobalModel, BlackBoxLearner, Array}, dir::String)

Saves IAI fits associated with different OptimalConstraintTree objects.
"""
function save_fit(bbl::BlackBoxLearner, dir::String = TREE_DIR)
    # TODO: support saving all of the BBR learner history. 
    IAI.write_json(dir * bbl.name * ".json", bbl.learners[end]) # save learner
    if bbl isa BlackBoxClassifier
        save(dir * bbl.name * ".jld", Dict("learner_kwargs" => bbl.learner_kwargs[end],
                                        "accuracies" => bbl.accuracies[end]))
    else
        save(dir * bbl.name * ".jld", Dict("learner_kwargs" => bbl.learner_kwargs[end],
                                        "thresholds" => bbl.thresholds[end],
                                        "ul_data" => bbl.ul_data[end]))
    end
    return
end

save_fit(bbls::Array, dir::String = TREE_DIR) = [save_fit(bbl, dir) for bbl in bbls]
save_fit(gm::GlobalModel, dir::String = TREE_DIR) = save_fit(gm.bbls, dir * "$(gm.name)_")

"""
    load_fit(BlackBoxLearner, dir::String = TREE_DIR)

Loads IAI fits associated with OptimalConstraintTree objects.
Checks that there is correspondence between loaded trees and the associated constraints.
"""

function load_fit(bbl::BlackBoxLearner, dir::String = TREE_DIR)
    loaded_lnr = IAI.read_json(dir * bbl.name * ".json");
    size(IAI.variable_importance(loaded_lnr), 1) == length(bbl.vars) || throw(
        OCTException("Object " * bbl.name * " does not match associated learner."))
    set_param(bbl, :reloaded, true)
    push!(bbl.learners, loaded_lnr)
    dd = load(dir * bbl.name * ".jld")
    if bbl isa BlackBoxClassifier
        push!(bbl.learner_kwargs, dd["learner_kwargs"])
        push!(bbl.accuracies, dd["accuracies"])
    else
        push!(bbl.learner_kwargs, dd["learner_kwargs"])
        push!(bbl.thresholds, dd["thresholds"])
        push!(bbl.ul_data, dd["ul_data"])
    end
    return
end

load_fit(bbls::Array{BlackBoxLearner}, 
        dir::String = TREE_DIR) = [load_fit(bbl, dir) for bbl in bbls]

load_fit(gm::GlobalModel, dir::String = TREE_DIR) = load_fit(gm.bbls, dir * "$(gm.name)_")
function plot(tree::IAI.GridSearch)
    """ Shows grid in browser. """
    try
        IAI.show_in_browser(tree.lnr)
    catch err
        if isa(err, UndefRefError)
        @warn("Certain trees are untrained.")
        else
            rethrow(err)
        end
    end
end

function show_trees(bbf::Union{BlackBoxFunction, DataConstraint})
    for grid in bbf.learners
        IAI.show_in_browser(grid.lnr)
    end
    return
end

function plot_accuracies(obj::Array{BlackBoxFunction, DataConstraint})
    bar(1:length(obj), [bbf.accuracies[end] for bbf in obj], xlabel="Constraint number",
                    xticks=1:length(ineqs), title="Constraint accuracies", legend=false)
end
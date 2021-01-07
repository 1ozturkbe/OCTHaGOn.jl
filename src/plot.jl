function show_trees(bbf::Union{BlackBoxFunction})
    for grid in bbf.learners
        IAI.show_in_browser(grid.lnr)
    end
    return
end

function plot_accuracies(obj::Array{BlackBoxFunction})
    bar(1:length(obj), [bbf.accuracies[end] for bbf in obj], xlabel="Constraint number",
                    xticks=1:length(ineqs), title="Constraint accuracies", legend=false)
end
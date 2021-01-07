function show_trees(bbf::Union{BlackBoxClassifier, BlackBoxRegressor})
    for grid in bbf.learners
        IAI.show_in_browser(grid.lnr)
    end
    return
end
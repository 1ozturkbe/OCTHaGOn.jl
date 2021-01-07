function show_trees(bbl::Union{BlackBoxClassifier, BlackBoxRegressor})
    for grid in bbl.learners
        IAI.show_in_browser(grid.lnr)
    end
    return
end
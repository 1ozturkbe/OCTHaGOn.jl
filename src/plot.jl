function show_trees(bbl::BlackBoxLearner)
    for lnr in bbl.learners
        IAI.show_in_browser(lnr)
    end
    return
end

show_trees(gm::GlobalModel) = show_trees.(gm.bbls)
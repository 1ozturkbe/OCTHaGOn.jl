function test_relaxations(gm::GlobalModel = minlp(true),
                          solver = CPLEX_SILENT)
    gm = minlp(true)
    set_optimizer(gm, solver)
    bound!(gm, gm.vars[end] => [-10,20])
    uniform_sample_and_eval!(gm)
    # Separate by percentiles of infeasible
    percentiles = [0.75, 0.9, 0.98, 1.0]
    bbf = gm.bbfs[1]
    for i=1:length(thresholds)
        learn_constraint!(bbf, validation_criterion = :sensitivity, threshold = thresholds[i])
    end
    @test true
end
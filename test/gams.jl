
""" Testing that problems are correctly imported, with some random checking. """
function gams_import_checks()
    filenums = [2.15, 2.16, 2.17, 2.18, 3.2, "3.10", 3.13, 3.15, 3.16, 3.18, 3.25]
    filenames = ["problem" * string(filenum) * ".gms" for filenum in filenums]
    gms = Dict()
    for filename in filenames
        try
            gms[filename] = GAMS_to_GlobalModel(GAMS_DIR, filename)
        catch
            throw(OCTException(filename * " has an import issue."))
        end
    end

    filename = "problem3.13.gms"
    gm =  GAMS_to_GlobalModel(GAMS_DIR, filename)
    x = gm.model[:x]
    @test length(gm.vars) == 8
    @test all(bound == [0,100] for bound in values(get_bounds(flat(gm.model[:x]))))
    @test length(gm.bbfs) == 13
    bound!(gm, Dict(var => [-300,300] for var in gm.vars))
    sample_and_eval!(gm.bbfs[1])
    learn_constraint!(gm.bbfs[1])

    return true
end

function recipe(gm)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, Gurobi.Optimizer)
    find_bounds!(gm, all_bounds=false)
    gm.settings[:ignore_accuracy] = true
    sample_and_eval!(gm, n_samples = 500)
    sample_and_eval!(gm, n_samples = 500)
    @info ("Sample feasibilities ", feasibility(gm))
    learn_constraint!(gm)
    @info("Approximation accuracies: ", accuracy(gm))
    save_fit(gm)
    globalsolve(gm)
end

function nonlinear_solve(gm::GlobalModel)
    nonlinearize!(gm)
    set_optimizer(gm, Ipopt.Optimizer)
    optimize!(gm)
end

@test gams_import_checks()

# filename = "problem3.13.gms"
# gm = GAMS_to_GlobalModel(GAMS_DIR, filename)
# bound!(gm, Dict(var => [-1000, 1000] for var in gm.vars))
# sample_and_eval!(gm);
# sample_and_eval!(gm);
# learn_constraint!(gm);
# set_optimizer(gm, Gurobi.Optimizer)
# gm.settings[:ignore_accuracy] = true
# globalsolve(gm)
# solution(gm)
#
# nonlinear_solve(gm)
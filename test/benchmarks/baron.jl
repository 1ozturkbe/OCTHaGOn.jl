include("../load.jl")

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    find_bounds!(gm, all_bounds=true)
    set_param(gm, :ignore_feasibility, true)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    @info ("Sample feasibilities ", feasibility(gm))
    learn_constraint!(gm)
    @info("Approximation accuracies: ", accuracy(gm))
    save_fit(gm)
    globalsolve(gm)
    return
end

# Loading BARON examples (that haven't been loaded yet)
for i in ["nlp1.jl", "nlp2.jl", "nlp3.jl"]
    include(OCT.BARON_DIR * i)
end


BARON_PROBLEMS = [gear, minlp, nlp1, nlp2, nlp3]

bms = [p(false) for p in BARON_PROBLEMS]
gms = [p(true) for p in BARON_PROBLEMS]

for bm in bms
    optimize!(bm);
end

for gm in gms
    try
        uniform_sample_and_eval!(gm, lh_iterations=5)
    catch
        println("$(gm.name): " * string(get_unbounds(gm)))
    end
end
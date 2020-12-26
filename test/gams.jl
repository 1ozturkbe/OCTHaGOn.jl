using Ipopt

""" Solver wrapper for GAMS benchmarking. """
function nonlinear_solve(gm::GlobalModel; solver = Ipopt.Optimizer)
    nonlinearize!(gm)
    set_optimizer(gm, solver)
    optimize!(gm)
end

""" Testing that problems are correctly imported, with some random checking. """
function gams_import_checks()
    @info("Testing gams imports.")
    filenums = [2.15, 2.16, 2.17, 2.18, 3.2, "3.10", 3.13, 3.15, 3.16, 3.18, 3.25]
    filenames = ["problem" * string(filenum) * ".gms" for filenum in filenums]
    gms = Dict()
    for filename in filenames
        try
            gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
#             println("    Problem NL constraints: " * string(length(gm.bbfs)))
            types = JuMP.list_of_constraint_types(gm.model)
            if !isempty(types)
                total_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
#                 println("    Problem total constraints: " * string(total_constraints))
            end
        catch
            throw(OCTException(filename * " has an import issue."))
        end
    end
    return true
end

function recipe(gm)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, Gurobi.Optimizer)
    find_bounds!(gm, all_bounds=true)
    gm.settings[:ignore_feasibility] = true
    gm.settings[:ignore_accuracy] = true
    sample_and_eval!(gm, n_samples = 500)
    sample_and_eval!(gm, n_samples = 500)
    @info ("Sample feasibilities ", feasibility(gm))
    learn_constraint!(gm)
    @info("Approximation accuracies: ", accuracy(gm))
    save_fit(gm)
    globalsolve(gm)
    return
end

@test gams_import_checks()

filename = "problem3.13.gms"
gm =  GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
x = gm.model[:x]
@test length(gm.vars) == 8
@test all(bound == [0,100] for bound in values(get_bounds(flat(gm.model[:x]))))
@test length(gm.bbfs) == 1

filename = "problem3.13.gms"
gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
bound!(gm, Dict(var => [-1, 1] for var in [gm.model[:f], gm.model[:y]]))
# bound!(gm, Dict(gm.model[:objvar] => [-40,-30]))
# recipe(gm)

set_optimizer(gm, Gurobi.Optimizer)
find_bounds!(gm, all_bounds=true)
bbf = gm.bbfs[1]
orig_bounds = get_bounds(bbf)
unbounds = get_unbounds(bbf)
@test length(unbounds) == 0
bounded_dict = Dict(key => val for (key, val) in orig_bounds if !any(isinf.(val)))

dir = "../OCTHaGOn_benchmarks/gams/global/"
filenames = readdir(dir)
GAMS_DIR = dir

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/gams.jl")
using BARON

pd = DataFrame()
n_max = 100
pd = CSV.read(dir * "problem_stats.csv", DataFrame)
pd = pd[pd.all_bounded .>= 0.5, :] # Use only benchmarks that are bounded...
pd = pd[pd.n_vars .<= n_max, :]    # and have fewer than n_max variables.

function gs_and_time!(m::Union{JuMP.Model, GlobalModel}, n_tries = 10)
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        set_param(m, :sample_coeff, 500)
        ct = 0
        set_param(m, :ignore_accuracy, true)
        set_param(m, :ignore_feasibility, true)
        while ct < n_tries
            try
                for bbl in m.bbls
                    if isempty(bbl.learners)
                        if !isempty(bbl.X)
                            bbl.X = DataFrame(string.(bbl.vars) .=> [Float64[] for i=1:length(bbl.vars)]) # Function samples
                            bbl.Y = []
                        end
                        uniform_sample_and_eval!(bbl)
                        if bbl isa BlackBoxClassifier
                            learn_constraint!(bbl, max_depth = 6)
                        elseif bbl isa BlackBoxRegressor
                            if length(bbl.vars) <= 6
                                learn_constraint!(bbl, max_depth = 6)
                            else
                                learn_constraint!(bbl, max_depth = 8, minbucket = 2*length(bbl.vars), 
                                hyperplane_config = (sparsity = 1,))
                            end
                        end
                    end
                    if isempty(bbl.mi_constraints) 
                        add_tree_constraints!(m, bbl)
                    end
                end
                ct += 1e5
            catch
                ct += 1
            end
        end
        if ct == n_tries
            throw(OCTHaGOnException("Globalsolve training failed for $(m.name)."))
        end    
        optimize!(m)
        descend!(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
    return
end

# Benchmarking loop
gms = Dict()
oct_results = DataFrame(:name => [], :oct_feas => [], :oct_cost => [], :oct_time => [])
for filename in pd.name[bench_idxs]
    @info "Trying $(filename)"
    gm = GAMS_to_GlobalModel(dir, filename * ".gms")
    set_optimizer(gm, SOLVER_SILENT)
    gms[gm.name] = gm
    t1 = time()
    try
        gs_and_time!(gm)
        set_param(gm, :tighttol, 1e-6)
        push!(oct_results, [gm.name,  is_feasible(gm), gm.cost[end], time() - t1])
    catch
        if !isempty(gm.cost)
            push!(oct_results, [gm.name,  is_feasible(gm), gm.cost[end], time() - t1])
        else
            push!(oct_results, [gm.name,  false, Inf, time() - t1])
        end
    end
end

# # Updating and rewriting data for OCTHaGOn results
# orig_data = CSV.read(dir * "problem_stats.csv", DataFrame)
# idxs = [idx for idx in 1:size(orig_data, 1) if orig_data.name[idx] in oct_results.name]
# orig_data[idxs,[:oct_feas, :oct_cost, :oct_time]] = oct_results[:,[:oct_feas, :oct_cost, :oct_time]]
# CSV.write(dir * "problem_stats.csv", orig_data)

baron_results = DataFrame(:name => [], :baron_feas => [], :baron_cost => [], :baron_time => [])
bms = Dict()
for filename in pd.name[bench_idxs]
    @info "Trying $(filename)"
    t1 = time()
    if filename in keys(bms)
        continue
    else
        m = GAMS_to_baron_model(dir, filename * ".gms", with_optimizer(BARON.Optimizer, PrLevel = 1, MaxTime = 410))
        bms[filename] = m
        try
            gs_and_time!(m)
            push!(baron_results, [filename,  true, objective_value(m), time() - t1])
        catch
            push!(baron_results, [filename,  false, Inf, time() - t1])
        end
    end
end

# # Updating and rewriting data for BARON results
# orig_data = CSV.read(dir * "problem_stats.csv", DataFrame)
# idxs = [idx for idx in 1:size(orig_data, 1) if orig_data.name[idx] in baron_results.name]
# orig_data[idxs,[:baron_feas, :baron_cost, :baron_time]] = baron_results[:, [:baron_feas, :baron_cost, :baron_time]]
# CSV.write(dir * "problem_stats.csv", orig_data)
dir = "../OCTHaGOn_benchmarks/gams/global/"
filenames = readdir(dir)
GAMS_DIR = dir

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/gams.jl")

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
        set_param(m, :sample_coeff, 400)
        ct = 0
        while ct < n_tries
            try
                globalsolve!(m) 
                ct += 1e5
            catch
                ct += 1
            end
        end
        if ct == n_tries
            throw(OCTHaGOnException("Globalsolve failed for $(m.name)."))
        end
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
    return
end

# Benchmarking loop
gms = Dict()
results = DataFrame(:name => [], :oct_feas => [], :oct_cost => [], :oct_time => [])
for filename in pd.name
    @info "Trying $(filename)"
    gm = GAMS_to_GlobalModel(dir, filename * ".gms")
    set_optimizer(gm, SOLVER_SILENT)
    gms[gm.name] = gm
    t1 = time()
    try
        gs_and_time!(gm)
        push!(results, [gm.name,  is_feasible(gm), gm.cost[end], time() - t1])
    catch
        if !isempty(gm.cost)
            push!(results, [gm.name,  is_feasible(gm), gm.cost[end], time() - t1])
        else
            push!(results, [gm.name,  false, Inf, time() - t1])
        end
    end
end
results.name = [na[1:end-4] for na in results.name] # Trimming the .gms string

# Updating and rewriting data
orig_data = CSV.read(dir * "problem_stats.csv", DataFrame)
idxs = [idx for idx in 1:size(orig_data, 1) if orig_data.name[idx] in results.name]
orig_data[idxs,[:oct_feas, :oct_cost, :oct_time]] = results[:,[:oct_feas, :oct_cost, :oct_time]]
CSV.write(dir * "problem_stats.csv", orig_data)

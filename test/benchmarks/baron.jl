include("../load.jl")

# function recipe(gm::GlobalModel)
#     t = [time()]
#     println("GlobalModel " * gm.name * " in progress...")
#     set_optimizer(gm, CPLEX_SILENT)
#     find_bounds!(gm, all_bounds=true)
#     set_param(gm, :ignore_feasibility, true)
#     set_param(gm, :ignore_accuracy, true)
#     push!(t, time())
#     uniform_sample_and_eval!(gm)
#     push!(t, time())
#     println("Sampling time: $(t[end])")
#     println("Sample feasibilities ", feasibility(gm))
#     learn_constraint!(gm)
#     push!(t, time())
#     print_ln("Approximation accuracies: ", accuracy(gm))
#     save_fit(gm)
#     push!(t, time())
#     globalsolve(gm)
#     push!(t, time())
#     return gm
# end

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    find_bounds!(gm, all_bounds=false)
    set_param(gm, :ignore_feasibility, true)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    save_fit(gm)
    globalsolve(gm)
end

# Loading BARON examples (that haven't been loaded yet)
for i in ["nlp1.jl", "nlp2.jl", "nlp3.jl"]
    include(OCT.BARON_DIR * i)
end

BARON_PROBLEMS = [gear, minlp, nlp1, nlp2, nlp3]

function optimize_and_time!(m::Union{JuMP.Model, GlobalModel})
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        recipe(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
end

bms = [p(false) for p in BARON_PROBLEMS]
optimize_and_time!.(bms)

gms = [p(true) for p in BARON_PROBLEMS]
optimize_and_time!.(gms)

# save_fit.(gms)
# load_fit.(gms)

# function regress_within_leaves(bbr::BlackBoxRegressor,  ignore_feas::Bool = false, interval::Float64 = 0.95; kwargs...)
#     check_sampled(bbr)
#     set_param(bbr, :reloaded, false) # Makes sure that we know trees are retrained. 
#     lnr = base_classifier()
#     IAI.set_params!(lnr; classifier_kwargs(; kwargs...)...)
#     threshold = quantile(bbr.Y, 1-interval)
#     nl = learn_from_data!(bbr.X, bbr.Y .<= threshold, lnr; fit_classifier_kwargs(; kwargs...)...) 
#     # SAMPLINGGGGG
#     # KNN SAMPLE OUTWARD
#     # IF SAMPLES FEASIBLE
#     # ALSO SAMPLE WITHIN LEAVES            
#     push!(bbr.learners, nl);
#     bbr.predictions = IAI.predict(nl, bbr.X)
#     push!(bbr.accuracies, IAI.score(nl, bbr.X, bbr.Y)) # TODO: add ability to specify criterion. 
#     push!(bbr.learner_kwargs, Dict(kwargs))
#     return
# end

include("../load.jl")


REPAIR = true 
OPT_SAMPLING = false
EQUALITIES = false 

using Serialization: serialize, deserialize
using DataFrames, Dates, LinearAlgebra, Random, BARON, Pkg
using Base

gm = nothing

"""
Loads and parses gam from file. 
If already loaded once, it will be loaded
from pickle instead of being parsed again.
"""
function load_gam_from_path(path; force_reload=false)

    if !force_reload && isfile("$(path).bin")
        gams = open("$(path).bin", "r") do io
            deserialize(io)
        end
        return gams
    end
    gams = GAMSFiles.parsegams(path)
    GAMSFiles.parseconsts!(gams)

    open("$(path).bin", "w") do io
        serialize(io, gams)
    end

    return gams 
end

function create_negative_definite_matrix(N)
    # Create a random matrix
    A = rand(N, N)

    # Create an orthogonal matrix
    U = eigvecs(A*A')

    # Create a negative-definite matrix Q
    Q = U * (Diagonal(rand(N)) .- 0.5) * U'
    
    return Q
end


"""
Creates a non-convex QP model with different numbers of variables/constraints.
Returns  a GlobalModel and a JuMP baron model 
"""

"""
Creates a non-convex QP model with different numbers of variables/constraints.
Returns  a GlobalModel and a JuMP baron model 
"""
function create_noncvx_qp_models(n_vars, n_constr, algs, linear_objective=true, relax_coeff=0; seed=123)
    
    Random.seed!(seed)
    
    N = n_vars
    M = n_constr

    
    # Create upper and bounds (those particular values are just an experiment)
    lbs = push!(-5*ones(N), -800)
    ubs = push!(5*ones(N), 0)
    
    local function create_gm_model(obj_expr, constr_expr, data)
        
        unrelaxed_obj = nothing 
        

        # Initialize JuMP model
        m = JuMP.Model(with_optimizer(CPLEX_SILENT))
        relax_var = @variable(m, r_rel>=0)
        relax_term = relax_coeff*relax_var;

        @variable(m, x[1:N+1])

        unrelaxed_obj = x[N+1]
        @objective(m, Min, unrelaxed_obj+relax_term)

        # Initialize GlobalModel
        gm = OCTHaGOn.GlobalModel(model = m, name = "qp")
        gm.objective = unrelaxed_obj
        gm.relax_var = relax_var 

        # Set bounds
        lbs_dict = Dict(x .=> lbs)
        ubs_dict = Dict(x .=> ubs)
        OCTHaGOn.bound!(gm, Dict(var => [lbs_dict[var], ubs_dict[var]] for var in gm.vars if var != relax_var))
        
        # Add objective
        # @TODO: see why expr_vars argument is needed
        OCTHaGOn.add_nonlinear_or_compatible(gm, 
            obj_expr, vars=x[1:N], 
            expr_vars=[x[1:N]], dependent_var = x[N+1],
            is_objective=true
        )
        
        #hashes = Set([])

        # Add constraints
        for (constr, params) in zip(constr_expr, data)
            global Q, c, d = params
            #println("MODEL CONSTR: ", constr)
            OCTHaGOn.add_nonlinear_constraint(gm, constr, vars=x[1:N], expr_vars=[x[1:N]], alg_list=algs, equality=EQUALITIES, regression=false)
            #constr_hash = bytes2hex(sha1("$(repr(constr))$(EQUALITIES)"))
            #push!(hashes, constr_hash)
        end
        
        OCTHaGOn.set_param(gm, :ignore_accuracy, true)
        #OCTHaGOn.set_param(gm, :ignore_feasibility, true)
        
        return gm
    end
    
    local function create_baron_model(obj_expr, constr_expr, data)
        
        global m = JuMP.Model(with_optimizer(BARON.Optimizer, PrLevel=0, MaxTime=120))
        global x = @variable(m, x[1:N])
        
        # Add objective
        @objective(m, Min, Base.invokelatest(eval(obj_expr), x))
        
        # Add upper/lower bound constraints
        @constraint(m, lbs[1:N] .<= x .<= ubs[1:N])
        
        # Add constraints
        for (constr, params) in zip(constr_expr, data)
            #println("BARON CONSTR: ", constr)
            global Q, c, d = params
            
            if EQUALITIES
                #@constraint(m, Base.invokelatest(eval(constr), x)==0)
                eval(Meta.parse("@NLconstraint(m, $(constr)==0)"))
            else 
                #@constraint(m, Base.invokelatest(eval(constr), x)>=0)
                #println(full_str)
                eval(Meta.parse("@NLconstraint(m, $(constr)>=0)"))
            end
        end
        
        return m
    end
    
    # Create the expression for the objective depending on whether
    # it is linear or not
    Q = create_negative_definite_matrix(N)
    c = randn(N)
    d = 0.1*randn()
    obj_expr = linear_objective ? :((x) -> $(c)'*x) : :((x) -> -x'*$(Q)*x - $(c)'*x)
    
    data = []

    if !linear_objective
        push!(data, (Q, c, d))
    end
    
    # Create the expressions for the constraints
    constr_expr = []
    constr_str = []
    for i=1:M
        Q = create_negative_definite_matrix(N)
        c = randn(N)
        d = 0.1*randn()
        #expr = :((x) -> -x'*$(Q)*x - $(c)'*x)
        expr = :((x) -> 0.5-1/(1+exp(-x'*$(Q)*x - $(c)'*x-$(d))))
        
        qp_expr = "-sum(x[i]*x[j]*Q[i,j] for i=1:$(N),j=1:$(N)) - sum(x[i]*c[i] for i=1:$(N))-d"
        full_expr = "0.5-1/(1+exp($(qp_expr)))"
        #expr_s = "@NLconstraint(m, 0.5-1/(1+exp($(qp_expr)))>=0)"
        
        push!(constr_expr, expr)
        push!(constr_str, full_expr)
        push!(data, (Q, c, d))
    end
    #return obj_expr
    return create_gm_model(obj_expr, constr_expr, data), create_baron_model(obj_expr, constr_str, data)
end


function solve_baron(gb, gm_hash, N)

    baron_output_path = "dump/qp/baron/baron_qp_nn.csv"

    try
        df = DataFrame(CSV.File(baron_output_path))
    catch 
        df = DataFrame()
    end

    d = Dict(row["hash"] => row["obj"] for row in eachrow(df))

    obj = nothing
    x = nothing
    # If the same problem has been solved by baron,
    # load the results from file
    if haskey(d, gm_hash)
        obj = d[gm_hash]
        println("Loading baron solution from file")
    else
        print("Solving using baron")
        optimize!(gb)
        print("Solved using baron")
        x_ref = [JuMP.variable_by_name(gb, "x[$(i)]") for i=1:N]
        x = value.(x_ref)
        obj = JuMP.objective_value(gb)

        df = append!(df, DataFrame("obj" => obj, "hash" => gm_hash))
        try
            CSV.write(baron_output_path, df)
        catch
            println("Couldn't write to CSV1")
        end 
    end
    return obj, x
end

"""
Returns a CSV with the stats of different
gam problems located in a subfolder of the fwolderr OCTHaGOn.GAMS_DIR.
Those sats include:
- Number of constraints 
- Number of variables 
- Number of bounded variables 
- Number of variables of each type (i.e. continuous, integer, binary)

The argument force_reload decides whether we should recreate the stats.
"""
function get_problem_stats(folder_name; force_reload=false, keep_only_selected=true)

    dir = OCTHaGOn.GAMS_DIR*folder_name*"\\"
    csv_path = dir*"problem_stats.csv"

    if !force_reload && isfile(csv_path)
        df = DataFrame(CSV.File(csv_path))
        if keep_only_selected
            df = filter(:selected => x -> x==1, df)
            if size(df, 1) == 0
                println("You have not selected any problems in the csv.")
                println("Please go to $(csv_path) and set selected=1 in the problems you want to select")
            end
        end
        return df
    end

    # Read all file names/paths in the folder
    all_paths = [d for d in readdir(dir; join=true) if occursin(".gms", d)]
    all_names = [replace(f, ".gms" => "") for f in readdir(dir; join=false) if occursin(".gms", f)]


    df = DataFrame()

    for (name, path) in zip(all_names, all_paths)

        try
            println("Adding $(name)")

            gams = load_gam_from_path(path; force_reload=false)

            data = DataFrame(
                "name" => name,
                "n_constr" => length(gams["equations"]),
                "n_vars" => 0,
                "n_bounded_vars" => 0,
                "continuous" => 0,
                "integer" => 0,
                "binary" => 0,
                "folder" => folder_name,
                "selected" => 0,
                "all_bounded" => 0,
                "optimal" => ""
            )
            
            vars = GAMSFiles.getvars(gams["variables"])

            for (var, info) in vars
                
                assignments = info.assignments
                type = info.typ # Variable type (free, positive, negative, integer, binary)
                #println(var)
                if var == "objvar" continue end 

                # The types of bounds our variable has 
                bound_types = Set([a.first.text for a in assignments if a.first.text ∈ ["lo","up"]])

                # Do some bookeeping to count the number of variables (and bounded variables)
                if type == "free"

                    data[1, "continuous"] += 1

                elseif type == "positive"
                
                    data[1, "continuous"] += 1
                    push!(bound_types,  "lo")

                elseif type == "negative"

                    data[1, "continuous"] += 1
                    push!(bound_types,  "lo")

                elseif type == "binary"
                    
                    data[1, "binary"] += 1
                    push!(bound_types,  "lo")
                    push!(bound_types,  "up")
                else
                    data[1, "integer"] += 1
                end
                
                if length(bound_types) >= 2
                    data[1, "n_bounded_vars"] += 1
                end
                data[1, "n_vars"] += 1
            end
            data[1, "all_bounded"] = 1*(data[1, "n_bounded_vars"] == data[1, "n_vars"])
            append!(df, data)
        catch
            println("Couldn't add file $(name)")
        end
    end

    CSV.write(csv_path, df)

    if keep_only_selected
        @warn("You have not selected any problems in the $(folder_name) csv.\n
        Please go to $(csv_path) and set selected=1 in the problems you want to select")
    end
    return df
end

function solve_and_benchmark(folders; alg_list = ["GBM", "SVM"])
    
    # function create_gm(name, folder)
    #     gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR*"$(folder)\\", name*".gms"; alg_list = alg_list, regression=false, relax_coeff=0)
    #     set_optimizer(gm, CPLEX_SILENT)
    #     set_param(gm, :sample_coeff, 1800)
    #     return gm 
    # end

    function solve_gm(gm; relax_coeff=0, ro_factor=0)
        
        set_param(gm, :ro_factor, ro_factor)
        gm.relax_coeff = relax_coeff

        globalsolve!(gm; repair=REPAIR, opt_sampling=OPT_SAMPLING)
        #feas_gap(gm)
        # Performance of the different algorithms (e.g. GBM, SVM, OCT)
        df_algs = vcat([bbl.learner_performance for bbl in gm.bbls]...)
        
        return df_algs, gm.cost[end], gm
    
    end

    # function solve_baron(name, folder)
    #     m = GAMS_to_baron_model(OCTHaGOn.GAMS_DIR*"$(folder)\\", name*".gms")
    #     optimize!(m)

    #     return JuMP.objective_value(m), DataFrame()
    # end

    df_all = DataFrame()
    df_algs_all = DataFrame()

    output_path = "dump/benchmarks/qp/"
    Base.Filesystem.mkpath(output_path)
    suffix = Dates.format(Dates.now(), "YY-mm-dd_HH-MM-SS")
    
    og_alg_list = copy(alg_list)

    for N = [5, 10, 20]
        for M = [1, 2, 3]
            id = N*1000+M
            alg_list = copy(og_alg_list)

            global gm, gb = create_noncvx_qp_models(N, M, alg_list;seed=1)
            gm_hash = OCTHaGOn.calculate_hash(gm)

            #global gm = create_gm(name, folder)
            id = 1
            solved = false 
            
            baron_obj,_ = solve_baron(gb, gm_hash, N)

            for ro_factor in [0.0]#[0.0,0.01,0.1,0.5,1]
                for hessian in [false]#, true
                    for momentum in [0., 0.8]
                        for relax_coeff in [0.0,1e2,1e4] #[0.0,1e2,1e4]
                            if solved 
                                continue
                            end

                            n_bbls = length([bbl for bbl in gm.bbls if (bbl isa BlackBoxClassifier || bbl isa BlackBoxRegressor)])
                            
                            #baron_obj = parse(Float32, replace(row["optimal"], r"[^0-9\.-]" => ""))
                            df_tmp = DataFrame(
                                "id"=> gm_hash,
                                "n_vars"=> N,
                                "n_constr"=> M,
                                "gm" => NaN, 
                                "baron" => baron_obj,
                                "diff" => NaN,
                                "subopt_factor" => NaN,
                                "gm_time" => NaN,
                                "ba_time" => NaN,
                                "algs" => "[\""*join(alg_list, "\",\"")*"\"]",
                                "feas_gaps" => [[]],
                                "ro_factor" => ro_factor,
                                "solved" => NaN,
                                "relax_coeff" => relax_coeff,
                                "n_bbls" => n_bbls,
                                "relax_epsilon" => NaN,
                                "momentum" => NaN,
                                "hessian" => false
                            )
                            
                            id += 1

                            try 
                                ts = time()
                                Random.seed!(50)

                                # relax_coeff = 0
                                df_algs = nothing 
                                gm_obj = nothing
                                #gm = nothing
                                # try
                                #     df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=0)
                                # catch
                                #     @info("Trying with relax var")
                                #     use_relax_var = true
                                #     df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=1)
                                # end
                                # gm_obj, df_algs = solve_baron(name, folder)
                                
                                #df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=relax_coeff)
                                
                                set_param(gm, :momentum, momentum)
                                set_param(gm, :second_order_repair, hessian)
                                df_algs, gm_obj, gm = solve_gm(gm; ro_factor=ro_factor, relax_coeff=relax_coeff)


                                gm_time = time()-ts
                                baron_time = gm_time
                                subopt = abs(baron_obj)<1 ? ((gm_obj+1)/(1+baron_obj)) : gm_obj/baron_obj
                                subopt = abs(baron_obj)<1 ? ((gm_obj-baron_obj)/(1+abs(baron_obj))) : (gm_obj-baron_obj)/abs(baron_obj)
                                subopt = 1-subopt
                                
                                feas_gaps = [bbl.feas_gap[end] for bbl in gm.bbls if isa(bbl, BlackBoxClassifier)]
                                # feas_gaps = []
                                
                                if abs(1-subopt) <= 1e-3  
                                    solved = true
                                end

                                df_tmp[!, "gm"] = [gm_obj]
                                df_tmp[!, "diff"] = [gm_obj-baron_obj]
                                df_tmp[!, "subopt_factor"] = [subopt]
                                df_tmp[!, "gm_time"] = [gm_time]
                                df_tmp[!, "ba_time"] = [gm_time]
                                df_tmp[!, "feas_gaps"] = [feas_gaps]
                                df_tmp[!, "solved"] = [1]
                                df_tmp[!, "relax_coeff"] = [relax_coeff]
                                df_tmp[!, "relax_epsilon"] = [gm.relax_epsilon]
                                df_tmp[!, "momentum"] = [get_param(gm, :momentum)]
                                df_tmp[!, "hessian"] = [get_param(gm, :second_order_repair)]

                                new_row = df_tmp #hcat(df_tmp, DataFrame(row))
                                append!(df_all, new_row)
                                append!(df_algs_all, df_algs)

                                println(df_all)

                                # break
                            catch e
                                df_tmp[!, "solved"] = [0]

                                new_row = df_tmp #hcat(df_tmp, DataFrame(row))
                                append!(df_all, new_row)
                                showerror(stdout, e)
                                #println("Error solving $(name)")
                                #println(stacktrace(catch_backtrace()))
                            end

                            try
                                csv_path = output_path*"benchmark$(suffix).csv"
                                csv_path_alg = output_path*"benchmark_alg$(suffix).csv"
                                #println(csv_path)
                                CSV.write(csv_path, df_all)
                                CSV.write(csv_path_alg, df_algs_all)
                            catch
                                println("Couldn't write to CSV")
                            end
                        end
                    end
                end
            end
        end
    end
    println(df_all)
end

folders = ["global"]

solve_and_benchmark(folders; alg_list = ["GBM", "SVM", "MLP"])
#"GBM", "SVM", "MLP"
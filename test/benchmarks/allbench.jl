
include("../load.jl")


REPAIR = true 
OPT_SAMPLING = false

using Serialization: serialize, deserialize
using DataFrames, Dates 


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

"""
Returns a CSV with the stats of different
gam problems located in a subfolder of the folderr OCTHaGOn.GAMS_DIR.
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
    
    function solve_gm(name, folder)
        
        gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR*"$(folder)\\", name*".gms"; alg_list = alg_list, regression=false)
        set_optimizer(gm, CPLEX_SILENT)
        set_param(gm, :sample_coeff, 1800)
        globalsolve!(gm; repair=REPAIR, opt_sampling=OPT_SAMPLING)
        
        # Performance of the different algorithms (e.g. GBM, SVM, OCT)
        df_algs = vcat([bbl.learner_performance for bbl in gm.bbls]...)
        

        return gm.cost[end], df_algs
    
    end

    function solve_baron(name, folder)
        m = GAMS_to_baron_model(OCTHaGOn.GAMS_DIR*"$(folder)\\", name*".gms")
        optimize!(m)

        return JuMP.objective_value(m), DataFrame()
    end

    df_all = DataFrame()
    df_algs_all = DataFrame()

    output_path = "dump/benchmarks/"
    Base.Filesystem.mkpath(output_path)
    suffix = Dates.format(Dates.now(), "YY-mm-dd_HH-MM-SS")
    
    og_alg_list = copy(alg_list)

    for folder in folders 
        df_stats = get_problem_stats(folder;force_reload=false)
        
        for (i, row) in enumerate(eachrow(df_stats))
            
            alg_list = copy(og_alg_list)

            name, folder = row["name"], row["folder"]
            println(name)
            # if name != "ex4_1_1" && name != "ex4_1_6"
            #     continue 
            # end
            if name ∉ ["ex8_3_14", "ex8_3_4", "ex5_2_5", "ex8_3_9", "ex8_3_3", "ex8_3_2"]
                continue
            end
            
            # if name ∉ ["ex8_3_14", "ex8_3_4", "ex5_2_5", "ex8_3_9", "ex8_3_3", "ex8_3_2", "ex5_4_4", "ex8_2_1a", "ex8_2_4a", "ex6_2_7", "ex6_2_5", "ex5_2_5", "ex5_3_3", "ex6_2_9", "ex6_2_10", "ex5_4_4", "ex6_2_13", "ex3_1_1", "ex7_2_3", "ex2_1_1", "ex7_2_4", "ex4_1_1"]
            #     continue
            # end
            try 
                ts = time()
                Random.seed!(50)
                gm_obj, df_algs = solve_gm(name, folder)
                #gm_obj, df_algs = solve_baron(name, folder)

                gm_time = time()-ts
                

                baron_obj = parse(Float32, replace(row["optimal"], r"[^0-9\.-]" => ""))
                baron_time = gm_time
                subopt = abs(baron_obj)<1 ? ((gm_obj+1)/1+baron_obj) : gm_obj/baron_obj
                subopt = abs(baron_obj)<1 ? ((gm_obj-baron_obj)/(1+abs(baron_obj))) : (gm_obj-baron_obj)/abs(baron_obj)
                subopt = 1-subopt
                df_tmp = DataFrame(
                    # "n" => N,
                    # "m" => M,
                    "gm" => gm_obj, 
                    "baron" => baron_obj,
                    "diff" => gm_obj-baron_obj,
                    "subopt_factor" => subopt,#gm_obj/baron_obj,
                    "gm_time" => gm_time,
                    "ba_time" => baron_time,
                    "algs" => "[\""*join(alg_list, "\",\"")*"\"]"
                )
                new_row = hcat(df_tmp, DataFrame(row))
                append!(df_all, new_row)
                append!(df_algs_all, df_algs)

                try
                    csv_path = output_path*"benchmark$(suffix).csv"
                    csv_path_alg = output_path*"benchmark_alg$(suffix).csv"
                    #println(csv_path)
                    CSV.write(csv_path, df_all)
                    CSV.write(csv_path_alg, df_algs_all)
                catch
                    println("Couldn't write to CSV")
                end

                println(df_all)

                # break
            catch e
                showerror(stdout, e)
                #println("Error solving $(name)")
                #println(stacktrace(catch_backtrace()))
            end
            # break
        end

        #break
    end
    println(df_all)
end

folders = ["global"]

solve_and_benchmark(folders; alg_list = ["GBM", "SVM", "MLP"])
#"GBM", "SVM", "MLP"
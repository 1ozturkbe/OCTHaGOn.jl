
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
    all_paths = readdir(dir; join=true)
    all_names = [replace(f, ".gms" => "") for f in readdir(dir; join=false)]


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
        
        gm = GAMS_to_GlobalModel(OCTHaGOn.GAMS_DIR*"$(folder)\\", name*".gms"; alg_list = alg_list)
        set_optimizer(gm, CPLEX_SILENT)
        globalsolve!(gm; repair=REPAIR, opt_sampling=OPT_SAMPLING)
        gm.cost[end]
    
    end

    df_all = DataFrame()

    output_path = "dump/benchmarks/"
    Base.Filesystem.mkpath(output_path)
    suffix = Dates.format(Dates.now(), "YY-mm-dd_HH-MM-SS")

    for folder in folders 
        df_stats = get_problem_stats(folder)
        
        for (i, row) in enumerate(eachrow(df_stats))

            try 
                name, folder = row["name"], row["folder"]
                ts = time()
                gm_obj = solve_gm(name, folder)
                gm_time = time()-ts
                

                baron_obj = parse(Float32, row["optimal"])
                baron_time = gm_time

                df_tmp = DataFrame(
                    # "n" => N,
                    # "m" => M,
                    "gm" => gm_obj, 
                    "baron" => baron_obj,
                    "diff" => gm_obj-baron_obj,
                    "subopt_factor" => baron_obj/gm_obj,
                    "gm_time" => gm_time,
                    "ba_time" => baron_time,
                    "algs" => "[\""*join(alg_list, "\",\"")*"\"]"
                )
                new_row = hcat(df_tmp, DataFrame(row))
                append!(df_all, new_row)

                try
                    csv_path = output_path*"benchmark$(suffix).csv"
                    #println(csv_path)
                    CSV.write(csv_path, df_all)
                catch
                    println("Couldn't write to CSV")
                end

                println(new_row)
            catch
                println("Error solving $(name)")
                println(stacktrace(catch_backtrace()))
            end
            #break
        end

        #break
    end
    println(df_all)
end

folders = ["global"]

solve_and_benchmark(folders; alg_list = ["OCT"])
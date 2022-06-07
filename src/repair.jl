function get_feasibilities(gm::GlobalModel, X; epsilon=1e-2)
    
    all_gaps = zeros(size(X,1))
    obj_feasible = zeros(size(X,1))
    
    for bbl in vcat(gm.bbls, gm.cfcs)
        
        gaps = []
        
        Y = bbl(X)
        
        if (bbl isa BlackBoxClassifier || bbl isa ClosedFormConstraint) && !isnothing(bbl.constraint)
            
            if bbl.equality
                all_gaps += 1.0*(abs.(Y) .<= epsilon)
            else
                all_gaps += 1.0*(Y .>= -epsilon)
            end
            
        elseif bbl isa BlackBoxRegressor && !isnothing(bbl.constraint)

            if isnothing(bbl.dependent_var)
                all_gaps += 1.0*(abs.(Y) .<= epsilon)
            else 
                obj_feasible = 1.0*(abs.(Y.-X[!,string(gm.objective)]) .<= epsilon)
                all_gaps += obj_feasible
            end
            
        end
        
        #push!(all_gaps, gaps)
    end
    all_gaps = all_gaps.*obj_feasible
    
    X[!, "num_feasibles"] = all_gaps
    return X
end

function get_best_feasible_objective(gm::GlobalModel, X::Union{DataFrame, DataFrameRow})
    
    X_f = get_feasibilities(gm, X)


    max_feas = maximum(X_f[:,"num_feasibles"])
    X_f = X_f[X_f[:,"num_feasibles"] .== max_feas,:];
    X_f = sort(X_f, "objvar", rev=false)

    best_row = copy(first(X_f,1))[!,string.(gm.vars)]
    
    return best_row
end

function final_sample_repair!(gm::GlobalModel)

    obj_bbl = filter(x -> x.dependent_var == gm.objective, 
                                [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor])


    sample_multiplier = 5;
    bounds = get_bounds(gm.vars);
    bounds = Dict(string(key)=>value for (key, value) in bounds)
    total_samples = sample_multiplier*size(obj_bbl[1].X,1)
    samples = Dict(key => rand(Uniform(b[1],b[2]),total_samples) for (key,b) in bounds)
    samples = DataFrame(samples)

    i = 1
    for (x,y) in zip(eachrow(copy(obj_bbl[1].X)), eachrow(copy(obj_bbl[1].Y))) 
        for j in 1:sample_multiplier
            samples[i,names(x)] = x
            samples[i,string(gm.objective)] = y[1]
            i += 1
        end
    end

    best_row = get_best_feasible_objective(gm, samples)
    last_id = length(gm.cost)

    append!(gm.solution_history, best_row);
    push!(gm.cost, best_row[1, string(gm.objective)]);

    # if length(gm.bbls) > 1

    try
        descend!(gm; append_x0=false)
        cur_id = length(gm.cost)

        idxes = 1:cur_id
        #idxes = filter((x)-> x !=last_id +1, last_id:cur_id)
        
        best_row = get_best_feasible_objective(gm, gm.solution_history[idxes, :])

        append!(gm.solution_history, best_row);
        push!(gm.cost, best_row[1, string(gm.objective)]);

    catch
        gm.solution_history = gm.solution_history[1:last_id-1, :]
        gm.cost = gm.cost[1:last_id-1]
    end
    
    return
end
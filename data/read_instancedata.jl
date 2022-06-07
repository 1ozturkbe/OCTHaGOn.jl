using CSV
using DataFrames

pd = CSV.read("instancedata.csv", DataFrame)

# Populating optimal solutions within minlp
minlp_pd = CSV.read("gams/minlp/problem_stats.csv", DataFrame)
col = Real[]
for problem_name in minlp_pd.name
    val = pd[pd.name .== problem_name, :primalbound]
    if isempty(val) || ismissing(val[1])
        @warn problem_name * " doesn't have a known optimum."
        push!(col, Inf)
    else
        push!(col, pd[pd.name .== problem_name, :primalbound][1])
    end
end

# Updating the CSV
if !hasproperty(minlp_pd, :optimum)
    insertcols!(minlp_pd, :optimum => col)
else
    minlp_pd.optimum = col
end

# Populating optimal solutions within global
global_pd = CSV.read("gams/global/problem_stats.csv", DataFrame)
col = Real[]
for problem_name in global_pd.name
    val = pd[pd.name .== problem_name, :primalbound]
    if isempty(val) || ismissing(val[1])
        @warn problem_name * " doesn't have a known optimum."
        push!(col, Inf)
    else
        push!(col, pd[pd.name .== problem_name, :primalbound][1])
    end
end

# Updating the CSV
if !hasproperty(global_pd, :optimum)
    insertcols!(global_pd, :optimum => col)
else
    global_pd.optimum = col
end

# Writing the optima back to the CSV
CSV.write("gams/minlp/problem_stats.csv", minlp_pd)
CSV.write("gams/global/problem_stats.csv", global_pd)
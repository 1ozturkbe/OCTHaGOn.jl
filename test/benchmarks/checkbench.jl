using DataFrames, CSV, Plots

dir = "../OCTHaGOn_benchmarks/gams/global/"
filenames = readdir(dir)
GAMS_DIR = dir

df = DataFrame()
n_max = 100
df = CSV.read(dir * "problem_stats.csv", DataFrame)
df = df[df.all_bounded .>= 0.5, :]
df = df[df.n_vars .<= n_max, :]
# df = df[1:93, :]

globalresults_idxs = 1:size(df, 1) #1:93 #[i for i = 1:93 if !isinf(df.optimum[i])]

function compute_error(i::Int64, df::DataFrame; method::String = "OCTHaGOn")
    error = 0
    optimum = df.oct_cost[i]
    true_optimum = df.optimum[i]
    if method == "OCTHaGOn"
    elseif method == "BARON"
        optimum = df.baron_cost[i]
    else
        throw(ErrorException("Method $(method) is not supported."))
    end
    if !isinf(true_optimum) && optimum isa Real
        if abs(true_optimum) <= 1
            return optimum - true_optimum
        else 
            return (optimum - true_optimum) / abs(true_optimum)
        end
    elseif !isinf(true_optimum) && isnan(optimum)
        return Inf
    else
        throw(ErrorException("Problem $(df.name[i]) doesn't have a proper global optimum. "))
    end
end

function return_feasibility(i::Int64, df::DataFrame; method::String = "OCTHaGOn")
    if method == "OCTHaGOn"
        return df.oct_feas[i]
    elseif method == "BARON"
        return df.baron_feas[i]
    else
        throw(ErrorException("Methoed $(method) is not supported."))
    end
end

# Set the desired method here
method = "BARON"

# Optimality plot
ylims = (-0.05, 0.35)
xlims = (0, 100)
plt = plot(ylims = ylims, xlims = xlims, xticks = 0:10:100, yticks = minimum(ylims):0.05:maximum(ylims), plot_title	= "Optimality gaps of global benchmarks")
xlabel!("Benchmark index")
ylabel!("Optimality gap")
feasibility = []
errors = []
for i = 1:length(globalresults_idxs)
    push!(feasibility, return_feasibility(i, df, method = method))
    push!(errors, compute_error(i, df, method = method))
end
feas_idxs = findall(x -> x == true, feasibility)
plt = scatter!(Array(1:length(globalresults_idxs))[feas_idxs], errors[feas_idxs], color = "blue", label = "feasible")
infeas_idxs = findall(x -> x == false, feasibility)
plt = scatter!(Array(1:length(globalresults_idxs))[infeas_idxs], errors[infeas_idxs], color = "red", label = "infeasible")
feas_quiver_idxs = intersect(findall(x -> x >= maximum(ylims), errors), findall(x -> x == true, feasibility))
plt = quiver!(Array(1:length(globalresults_idxs))[feas_quiver_idxs], 
    ones(length(feas_quiver_idxs)) * maximum(ylims) * 7/8, 
    quiver=(zeros(length(feas_quiver_idxs)), ones(length(feas_quiver_idxs)) * maximum(ylims)*1/8), 
    arrow = true, color = "blue", linewidth = 1)
infeas_quiver_idxs = intersect(findall(x -> x >= maximum(ylims), errors), findall(x -> x == false, feasibility))
plt = quiver!(Array(1:length(globalresults_idxs))[infeas_quiver_idxs], 
    ones(length(infeas_quiver_idxs)) * maximum(ylims) * 7/8, 
    quiver=(zeros(length(infeas_quiver_idxs)), ones(length(infeas_quiver_idxs)) * maximum(ylims)*1/8), 
    arrow = true, color = "red", linewidth = 1, legend = :outerright)
display(plt)

# Solution time plot
times = df.oct_time[globalresults_idxs]
n_vars = df.n_vars[globalresults_idxs]   
if method == "BARON"
    times = df.baron_time[globalresults_idxs]
end
ylims = (0, maximum(times)+10)
xlims = (0,100)
plt = plot(ylims = ylims, xlims = xlims, xticks = 0:10:100, yticks = minimum(ylims):100:maximum(ylims), 
    plot_title	= "Solution times of global benchmarks")
xlabel!("Number of variables")
ylabel!("Solution time (s)")
plt = scatter!(n_vars[infeas_idxs], times[infeas_idxs], color = "red", label = "infeasible", legend = :bottomright)
plt = scatter!(n_vars[feas_idxs], times[feas_idxs], color = "blue", label = "feasible", legend = :bottomright)
display(plt)

# # Checking that problems are imported properly...
# valid_filenames = []
# for filename in df.name
#     filename = filename * ".gms"
#     @info "Trying " * filename * "."
#     model = JuMP.Model()
#     # Parsing GAMS Files
#     lexed = GAMSFiles.lex(GAMS_DIR * filename)
#     gams = []
#     try
#         gams = GAMSFiles.parsegams(GAMS_DIR * filename)
#     catch e
#         if e isa KeyError
#             @warn filename * " failed due to KeyError."
#         elseif e isa LoadError
#             @warn filename * " failed due to an InvalidCharError. "
#         else
#             @warn filename * " failed due to an unknown Error."
#         end
#         continue
#     end
#     GAMSFiles.parseconsts!(gams)

#     vars = GAMSFiles.getvars(gams["variables"])
#     sets = Dict{String, Any}()
#     if haskey(gams, "sets")
#         sets = gams["sets"]
#     end
#     preexprs, bodyexprs = Expr[], Expr[]
#     if haskey(gams, "parameters") && haskey(gams, "assignments")
#         GAMSFiles.parseassignments!(preexprs, gams["assignments"], gams["parameters"], sets)
#     end

#     # Getting variables
#     vardict, constdict = generate_variables!(model, gams) # Actual JuMP variables
#     if length(vardict) <= n_max
#         @info filename * " added with $(length(vardict)) variables."
#         push!(valid_filenames, filename)
#     end
# end
using DataFrames, CSV, Plots

dir = "../OCTHaGOn_benchmarks/gams/global/"
filenames = readdir(dir)
GAMS_DIR = dir

pd = DataFrame()
n_max = 100
pd = CSV.read(dir * "problem_stats.csv", DataFrame)
pd = pd[pd.all_bounded .>= 0.5, :]
pd = pd[pd.n_vars .<= n_max, :]
pd = pd[1:80, :]

globalresults_idxs = [i for i = 1:80 if !isinf(pd.optimum[i])]

# Plotting the actual optimality gaps (NOTE: perhaps make a log plot?)
ylims = (-0.15, 0.35)
plt = plot(ylims = ylims, xticks = 0:10:80, yticks = minimum(ylims):0.05:maximum(ylims), plot_title	= "Optimality gaps of global benchmarks")
xlabel!("Problem number")
ylabel!("Optimality gap")
for i = 1:length(globalresults_idxs)
    color = "blue"
    if !pd.oct_feas[i] 
        color = "red"
    end
    error = 0
    if abs(pd.optimum[i]) <= 1
        error = pd.oct_cost[i] - pd.optimum[i]
    else
        error = (pd.oct_cost[i] - pd.optimum[i]) / pd.optimum[i]
    end
    if error >= maximum(ylims)
        plt = quiver!([i], [maximum(ylims)*7/8], quiver=([0], [maximum(ylims)*1/8]), arrow = true, color = color, linewidth = 1)
    elseif error <= minimum(ylims)
        plt = quiver!([i], [-maximum(ylims)*7/8], quiver=([0], [-maximum(ylims)*1/8]), arrow = true, color = color, linewidth = 1)
    else
        plt = scatter!([i], [error], color = color, legend = false)
    end
end
display(plt)

# # Checking that problems are imported properly...
# valid_filenames = []
# for filename in pd.name
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
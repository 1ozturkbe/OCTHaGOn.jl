include("../test/load.jl")

using DataFrames
using Test
using Plots
using CSV
using Gurobi

@info "Trains over axial flux motor data."

# Getting simulation data
X = CSV.read("data/afpm/afpm_inputs.csv",
                 copycols=true, delim=",");
X_infeas = CSV.read("data/afpm/afpm_infeas_inputs.csv",
                    copycols=true, delim=",");
Y = CSV.read("data/afpm/afpm_outputs.csv",
                 copycols=true, delim=",");
X = select!(X, Not(:Column1));
X_infeas = select!(X_infeas, Not(:Column1));
Y = select!(Y, Not(:Column1));

# Free variables
varkeys = Symbol.(names(X))
objkeys = Symbol.(names(Y))

# Creating model with relevant variables
m = Model(Gurobi.Optimizer)
inputs = [];
for key in varkeys
    push!(inputs, @variable(m, base_name = string(key)));
end
outputs = [];
for key in objkeys
    push!(outputs, @variable(m, base_name = string(key)));
end

# Bounding and Integer constraints for input variables
N_coils, TPC, p = inputs[4:6];
baseline = Dict(inputs[i] => [13, 7.6, 1., 18, 10, 16, 0.15, 3.0, 0.45][i] for i=1:length(varkeys));
ranges = Dict(key => [0.5*value, 1.5*value] for (key, value) in baseline);
ranges[inputs[9]] = [0.25*baseline[inputs[9]], 2.25*baseline[inputs[9]]]; # wire_A needs special care...
# ranges = Dict(key => log.(value) for (key, value) in ranges)
ranges = Dict(var => [log.(minimum(X[string(var)])),log.(maximum(X[string(var)]))] for var in inputs)

# Bounding for output variables of interest
idxs = [1, 4, 6, 8, 10, 12];
feas_idxs = findall(x -> x .>= 0.5, Y.Efficiency);
X_feas = X[feas_idxs, :];
Y_feas = Y[feas_idxs, :];
out_ranges = Dict(key => [minimum(log.(Y_feas[Symbol(key)])), maximum(log.(Y_feas[Symbol(key)]))] for key in outputs[idxs]);
bound!(m, ranges);

# Geometry constraints (in logspace)
D_out, D_in = inputs[1:2]
@constraint(m, D_out >= D_in)

N_coils_range = log.(unique(X_feas["N_coils"]));
int = @variable(m, [1:length(N_coils_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, N_coils == sum(N_coils_range .* int))

p_range = log.(unique(X_feas["p"]))
int = @variable(m, [1:length(p_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, p == sum(p_range .* int))
@constraint(m, N_coils >= p + 1e-5) # motor type 2

# Objectives and FOMs
output_idxs = [1, 4, 6, 8, 10, 12];
P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
# set_lower_bound(Efficiency, 0)
# set_upper_bound(Efficiency, 0)
@constraint(m, log.(9.8) <= P_shaft <= log.(10.2))
@constraint(m, log.(7800) <= Rotational_Speed <= log.(8200))

# Fitting power closure, and creating a global model
# simulation = DataConstraint(vars = inputs)
feasmap = zeros(size(Y, 1)); feasmap[feas_idxs] .= 1;
# add_data!(simulation, log.(X), feasmap)
# learn_constraint!(simulation)
# lnr = IAI.fit!(base_otc(), log.(X), feasmap)
# IAI.write_json("power_closure.json", lnr)
lnr = IAI.read_json("power_closure.json")
constrs, leaf_vars = add_feas_constraints!(m, inputs, lnr, M=1e3, return_data = true)
# simulation.mi_constraints = constrs; # Note: this is needed to monitor the presence of tree
# simulation.leaf_variables = leaf_vars; #  constraints and variables in gm.model

# Only one feasible leaf (4), so can just use regression equations
FOMs = [Mass_Specific_Power, Mass, Rotational_Speed, Efficiency]
leaf = 4;

# Regressions over leaves
leaf_index, all_leaves = bin_to_leaves(lnr, log.(X_feas))
for FOM in FOMs
    regressor = regress(log.(X_feas), log.(Y_feas[string(FOM)]))
    constant = IAI.get_prediction_constant(regressor)
    weights  = IAI.get_prediction_weights(regressor)[1]
    vks = Symbol.(names(X_feas))
    β = []
    for i = 1:size(vks, 1)
        if vks[i] in keys(weights)
            append!(β, weights[vks[i]])
        else
            append!(β, 0.0)
        end
    end
    @constraint(m, sum(β .* inputs) + constant == FOM)
end

# Solving
@objective(m, Max, Mass_Specific_Power)
optimize!(m)
println("Inputs")
for i=1:length(inputs)
    println(string(inputs[i], " ", exp(getvalue(inputs[i]))))
end
println("FOMs")
for FOM in FOMs
    println(string(FOM, " ", exp(getvalue(FOM))))
end

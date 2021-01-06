include("../test/load.jl")

using DataFrames
using Test
using CSV

@info "Trains over axial flux motor data."

# Getting simulation data
X = CSV.read("data/afpm/afpm_inputs.csv", DataFrame,
                 copycols=true, delim=",");
X_infeas = CSV.read("data/afpm/afpm_infeas_inputs.csv", DataFrame,
                    copycols=true, delim=",");
Y = CSV.read("data/afpm/afpm_outputs.csv", DataFrame,
                 copycols=true, delim=",");
X = select!(X, Not(:Column1));
X_infeas = select!(X_infeas, Not(:Column1));
Y = select!(Y, Not(:Column1));

# Free variables
varkeys = Symbol.(names(X))
objkeys = Symbol.(names(Y))

# Creating model with relevant variables
m = Model(with_optimizer(CPLEX_SILENT))
inputs = [];
for key in varkeys
    nv = @variable(m, base_name = string(key))
    m[Symbol(key)] = nv
    push!(inputs, nv);
end
outputs = [];
for key in objkeys
    nv = @variable(m, base_name = string(key))
    m[Symbol(key)] = nv
    push!(outputs, nv);
end

# Bounding and Integer constraints for input variables
N_coils, TPC, p = inputs[4:6];
# baseline = Dict(inputs[i] => [13, 7.6, 1., 18, 10, 16, 0.15, 3.0, 0.45][i] for i=1:length(varkeys));
# ranges = Dict(key => [0.5*val, 1.5*val] for (key, val) in baseline);
# ranges[inputs[9]] = [0.25*baseline[inputs[9]], 2.25*baseline[inputs[9]]]; # wire_A needs special care...
# ranges = Dict(key => log.(val) for (key, val) in ranges)
ranges = Dict(var => [log.(minimum(X[!, string(var)])),log.(maximum(X[!, string(var)]))] for var in inputs)

# Bounding for output variables of interest
idxs = [1, 4, 6, 8, 10, 12];
feas_idxs = findall(x -> x .>= 0.5, Y.Efficiency);
X_feas = X[feas_idxs, :];
Y_feas = Y[feas_idxs, :];
out_ranges = Dict(key => [minimum((Y_feas[!, Symbol(key)])), maximum((Y_feas[!, Symbol(key)]))] for key in outputs[idxs]);
bound!(m, ranges);

# Geometry constraints (in logspace)
D_out, D_in, N_coils, wire_w = inputs[1], inputs[2], inputs[4], inputs[7]
@constraint(m, D_out >= D_in)
@constraint(m, log(pi) + D_in >= log(0.2) + wire_w + N_coils) #pi*D_in >= 2*0.1*wire_w*N_coils

N_coils_range = log.(unique(X_feas[!, "N_coils"]));
int = @variable(m, [1:length(N_coils_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, N_coils == sum(N_coils_range .* int))

p_range = log.(unique(X_feas[!, "p"]))
int = @variable(m, [1:length(p_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, p == sum(p_range .* int))
@constraint(m, N_coils >= p + 1e-3) # motor type 2

TPC_range = log.(unique(X_feas[!, "TPC"]))
int = @variable(m, [1:length(TPC_range)], Bin)
@constraint(m, sum(int) == 1)
@constraint(m, TPC == sum(TPC_range .* int))

# Objectives and FOMs
output_idxs = [1, 4, 6, 8, 10, 12];
P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
set_upper_bound(Efficiency, 1)

# Fitting power closure, and creating a global model
feasmap = zeros(size(Y, 1)); feasmap[feas_idxs] .= 1;
# simulation = DataConstraint(xvars = inputs, yvars = nothing)
# simulation.X = log.(X)
# simulation.Y = feasmap
# add_data!(simulation, log.(X), feasmap)
# learn_constraint!(simulation)
# lnr = IAI.fit!(base_lnr(false), log.(X), feasmap)
# IAI.write_json("power_closure.json", lnr)
lnr = IAI.read_json("power_closure.json")
constrs, leaf_vars = add_feas_constraints!(m, inputs, lnr, M=1e3)
# simulation.mi_constraints = constrs; # Note: this is needed to monitor the presence of tree
# simulation.leaf_variables = leaf_vars; #  constraints and variables in gm.model

"""
Basic regression purely for debugging.
"""
function regress(points::DataFrame, values::Array; weights::Array = ones(length(values)))
    lnr= IAI.OptimalFeatureSelectionRegressor(sparsity = :all);
    IAI.fit!(lnr, points, values, sample_weight=weights)
    return lnr
end

# Only one feasible leaf (4), so can just use regression equations
# Regressions over leaves
P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
FOMs = [P_shaft, Rotational_Speed, Mass, Efficiency]
leaf_index, all_leaves = bin_to_leaves(lnr, log.(X_feas))
for FOM in FOMs
    regressor = regress(log.(X_feas), log.(Y_feas[!,string(FOM)]))
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
@constraint(m, log(9.8) <= P_shaft)
@constraint(m, log.(7800) <= Rotational_Speed <= log(8200))
@objective(m, Min, Mass)
optimize!(m)
println("Inputs")
for i=1:length(inputs)
    println(string(inputs[i], " ", exp(getvalue(inputs[i]))))
end
println("FOMs")
for FOM in FOMs
    println(string(FOM, " ", exp(getvalue(FOM))))
end

# fdf = DataFrame(names(X) .=> exp.(getvalue.(inputs)))
# CSV.write("data/afpm/afpm_opt.csv", fdf)
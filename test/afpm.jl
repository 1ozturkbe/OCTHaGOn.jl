using DataFrames
using Gurobi, JuMP
using Test
using Plots
using CSV

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree
global PROJECT_ROOT = @__DIR__

@info "Trains over axial flux motor data."
lnr = OCT.base_otr()
# Getting data from aerospace database.
X = CSV.read("../data/afpm/afpm_inputs.csv",
                 copycols=true, delim=",");
x_infeas = CSV.read("../data/afpm/afpm_infeas_inputs.csv",
                    copycols=true, delim=",")
Y = CSV.read("../data/afpm/afpm_outputs.csv",
                 copycols=true, delim=",");
X = select!(X, Not(:Column1))
Y = select!(Y, Not(:Column1))
valid_idxs = findall(x -> x .>= 0.5, Y.Efficiency);
X = X[valid_idxs,:];
Y = Y[valid_idxs,:];

# Learning geometry infeasibility
lnr = IAI.fit()

# Learning specific power
# lnr = IAI.fit!(OCT.gridify(lnr), log.(X), log.(Y[!, Symbol("Mass Specific Power")]))
# IAI.write_json(string(PROJECT_ROOT,"/test/data/afpm_specpower_learner.json"), lnr)
#
# Learning mass
# lnr = IAI.fit!(OCT.gridify(lnr), log.(X), log.(Y[!, Symbol("Mass")]))
# IAI.write_json(string(PROJECT_ROOT,"/test/data/afpm_mass_learner.json"), lnr)


specpower_learner = IAI.read_json(string(PROJECT_ROOT,"/data/afpm_specpower_learner.json"))
mass_learner = IAI.read_json(string(PROJECT_ROOT, "/data/afpm_mass_learner.json"))

IAI.show_in_browser(specpower_learner.lnr);
IAI.show_in_browser(mass_learner.lnr);
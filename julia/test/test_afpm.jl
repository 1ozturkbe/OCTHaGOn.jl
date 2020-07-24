using DataFrames
using Gurobi, JuMP
using Test
using Plots
using CSV

include("../src/OptimalConstraintTree.jl")
global OCT = OptimalConstraintTree
global PROJECT_ROOT = @__DIR__

function fit_afpmModel()
    @info "Trains over axial flux motor data."
    lnr = OCT.base_otr()
    # Getting data from aerospace database.
    vks = ["Re", "thick", "M", "C_L"];
    X = CSV.read("../data/airfoil/airfoil_X.csv",
                 copycols=true, header=vks, delim=",");
    Y = CSV.read("../data/airfoil/airfoil_y.csv",
                 copycols=true, header=["C_D"], delim=",");
        return lnr
end

@test fit_afpmModel()


#=
load:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-09-04
=#

using Pkg
Pkg.activate("test")

using DataFrames
using JuMP
using CPLEX
using CSV
# using Gurobi
# using Ipopt
using JLD
using MathOptInterface
using Test
using Random

include("../src/OCTHaGOn.jl")
using .OCTHaGOn
global MOI = MathOptInterface
global SOLVER_SILENT = with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0)
# global GUROBI_SILENT = with_optimizer(Gurobi.Optimizer, OutputFlag = 0, Gurobi.Env())
Random.seed!(1);
MOI.Silent() = true;

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/gams.jl");

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/sagemark.jl");

include(OCTHaGOn.PROJECT_ROOT * "/test/tools/models.jl")

include(OCTHaGOn.DATA_DIR * "speed_reducer.jl")

include(OCTHaGOn.DATA_DIR * "water_network.jl")

include(OCTHaGOn.DATA_DIR * "oos.jl")

include(OCTHaGOn.DATA_DIR * "afpm.jl")

include(OCTHaGOn.DATA_DIR * "polynomial.jl")

include(OCTHaGOn.DATA_DIR * "foxes_and_rabbits.jl")

include(OCTHaGOn.BARON_DIR * "gear.jl")

include(OCTHaGOn.BARON_DIR * "minlp.jl")

include(OCTHaGOn.BARON_DIR * "minlp_demo.jl")

include(OCTHaGOn.BARON_DIR * "nlp1.jl")

include(OCTHaGOn.BARON_DIR * "nlp2.jl")

include(OCTHaGOn.BARON_DIR * "nlp3.jl")

include(OCTHaGOn.BARON_DIR * "pool1.jl")
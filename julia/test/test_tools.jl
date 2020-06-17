#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests src/tools.jl.
=#

using Test
using MathOptInterface;
global MOI = MathOptInterface;
MOI.Silent()
include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;

# Solving a MOI model for comparison and bound generation
filename = "../../data/cblib.zib.de/flay03m.cbf.gz";
mof_model = OCT.CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
@test mof_obj ≈ 48.989798037382

# Importing CBF to ModelData using MathProgBase
filename = "../../data/cblib.zib.de/flay03m.cbf.gz";
md = OCT.CBF_to_ModelData(filename);
# Setting arbitrary bounds for unbounded problem
OCT.update_bounds!(md, mof_vars .- 40., mof_vars .+ 40.);

# Importing sagebenchmark to ModelData and checking it
# sagebench = OCT.sagemark_to_ModelData(3)
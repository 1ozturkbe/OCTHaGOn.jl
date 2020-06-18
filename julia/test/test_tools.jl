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
filename = "../../data/cblib.zib.de/clay0203h.cbf.gz";
mof_model = OCT.CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
@test mof_obj ≈ 41573.2611172422

# Importing CBF to ModelData using MathProgBase
md = OCT.CBF_to_ModelData(filename);
md.name = "clay0203h";
# Setting arbitrary bounds for unbounded problem
OCT.update_bounds!(md, mof_vars .- 50000., mof_vars .+ 50000.);
# Sampling ModelData
X = OCT.sample(md);
# Fitting model
ineq_trees, eq_trees = OCT.fit(md, X, lnr = OCT.base_otc(), write=true);


# Importing sagebenchmark to ModelData and checking it
# sagebench = OCT.sagemark_to_ModelData(3)
#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests src/tools.jl.
=#

using Test
using MathOptInterface;
using JuMP;
global MOI = MathOptInterface;
MOI.Silent() = true
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
OCT.update_bounds!(md, mof_vars .- 100., mof_vars .+ 100.);

# Sampling ModelData
# X = OCT.sample(md);
# Fitting model
# ineq_trees, eq_trees = OCT.fit(md, X, lnr = OCT.base_otc(), dir=string("data/",md.name));

jm, jx = OCT.jump_it(md); # Creating the JuMPModel
ineq_trees, eq_trees = OCT.import_trees(string("data/", md.name), md)
OCT.add_linear_constraints!(jm, jx, md);
OCT.add_tree_constraints!(jm, jx, ineq_trees, eq_trees);
JuMP.solve(jm);

# TODOs
# Document the structure to Dimitris.


# Importing sagebenchmark to ModelData and checking it
md = OCT.sagemark_to_ModelData(3, lse=true);
md.lbs[end] = -170;
md.ubs[end]= -120;
X = OCT.sample(md);
ineq_trees, eq_trees = OCT.fit(md, X, lnr = OCT.base_otc(), dir=string("data/",md.name));
jm, jx = OCT.jump_it(md);
OCT.add_linear_constraints!(jm, jx, md);
OCT.add_tree_constraints!(jm, jx, ineq_trees, eq_trees);
status = solve(jm);
println("Solved minimum: ", sum(md.c .* getvalue(jx)))
println("Known global bound: ", -147-2/3)
println("X values: ", getvalue(jx))
println("Optimal X: ", [5.01063529, 3.40119660, -0.48450710, -147-2/3])
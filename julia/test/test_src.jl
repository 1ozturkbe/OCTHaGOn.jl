#=
test_src:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code.
All the rest of the tests look at examples
=#

using Test
using Gurobi
using MosekTools
using JuMP
using MathOptInterface

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
const MOI = MathOptInterface
const PROJECT_ROOT = @__DIR__

# Initialization tests
md = OCT.ModelData(c = [1,2,3]);
@test md.lbs == -Inf.*ones(length(md.c))

# Check infeasible bounds for Model Data
OCT.update_bounds!(md, [-2,1,3], [Inf, 5, 6]);
@test_throws ArgumentError OCT.update_bounds!(md, [-2,1,3], [Inf, -1, 6]);

# Check sampling Model Data
@test_throws ArgumentError OCT.sample(md)

# Check creation of JuMP.Model() from ModelData
jm, jx = OCT.jump_it(md);

# Test CBF imports
filename = string("../data/cblib.zib.de/shortfall_20_15.cbf.gz");
mof_model = OCT.CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
# @test mof_obj ≈ -0.0777750720542

# ModelData
md = OCT.CBF_to_ModelData(filename);
md.name = "shortfall_20_15"
# OCT.find_bounds!(md);
md.lbs = zeros(length(md.c));
md.ubs = ones(length(md.c));

X = OCT.sample(md, n_samples=1000);
ineq_trees, eq_trees = OCT.fit(md, X)
m,x = OCT.jump_it(md, solver=GurobiSolver())
OCT.add_tree_constraints!(m, x, ineq_trees, eq_trees)
status = solve(m)
OCT_vars = getvalue(x)
OCT_obj = sum(md.c.*OCT_vars)
OCT.show_trees(ineq_trees)
err = (mof_vars - OCT_vars).^2
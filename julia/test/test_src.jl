#=
test_src:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code.
All the rest of the tests look at examples
=#

using Test
using MathOptInterface

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__

# Initialization tests
md = OCT.ModelData(c = [1,2,3]);

# Check infeasible bounds for Model Data
lbs = Dict(md.vks .=> [-2,1,3]);
ubs = Dict(md.vks .=> [Inf, 5, 6])
OCT.update_bounds!(md, lbs=lbs, ubs=ubs);
ubs = Dict(md.vks .=> [Inf, -1, 6])
@test_throws OCT.OCTException OCT.update_bounds!(md, ubs = ubs);

# Check sampling Model Data
@test_throws OCT.OCTException X = OCT.lh_sample(md, n_samples=100)

# # Check creation of JuMP.Model() from ModelData
OCT.jump_it!(md);
@test length(md.JuMP_model.linconstr) == 5;

# Test CBF imports
filename = string("../data/cblib.zib.de/shortfall_20_15.cbf.gz");
mof_model = OCT.CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
@test mof_obj ≈ -1.0792654303
#
# # Testing CBF import to ModelData
md = OCT.CBF_to_ModelData(filename);
md.name = "shortfall_20_15"
OCT.find_bounds!(md, all_bounds = true);

# Test sampling
n_samples = 500;
X = OCT.lh_sample(md, n_samples=n_samples);
#
# Testing constraint import.
bbf = md.fns[1];
Y = md.fns[1](X);

# Testing proper CBF constraint import.
using ConicBenchmarkUtilities
dat = readcbfdata(filename);
c, A, b, constr_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat);
(cone,idxs) = constr_cones[6];
var_idxs = unique(cart_ind[2] for cart_ind in findall(!iszero, A[idxs, :])); # CartesianIndices...
function constr_fn(x)
    let b = copy(b[idxs]), A = copy(A[idxs, var_idxs])
        expr = b - A*x;
        return expr[1].^2 - sum(expr[2:end].^2);
    end
end
Y2 = [constr_fn(Array(X[j,var_idxs])) for j=1:n_samples];
@test Y == Y2;

# Testing sample_and_eval for combined LH and boundary sampling.
OCT.sample_and_eval!(md, n_samples=500);
@test size(bbf.X) == (500,length(bbf.vks))
@test all(OCT.feasibility(md) .>= 0)

# Testing use KNN sampling and building trees
OCT.sample_and_eval!(md, n_samples=500);
OCT.learn_constraint!(md)

# Solving the model
status = OCT.solve(md)
OCT_vars = getvalue(md.JuMP_vars);

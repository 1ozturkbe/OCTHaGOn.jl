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

# Testing the SOC constraints.
Y = [md.ineq_fns[1](X[j,:]) for j=1:1000];
using ConicBenchmarkUtilities
dat = readcbfdata(filename);
c, A, b, constr_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat);
(cone,idxs) = constr_cones[6];
var_idxs = unique(cart_ind[2] for cart_ind in findall(!iszero, A[idxs, :])); # CartesianIndices...
function constr_fn(x)
    let b = copy(b[idxs]), A = copy(A[idxs, :])
        expr = b - A*x;
        return expr[1].^2 - sum(expr[2:end].^2);
    end
end
Y2 = [constr_fn(X[j,:]) for j=1:1000];
@test Y == Y2
println(sum(Y2))

# Original JuliaLang example
# a = [1,2,3,4,5];
# idxs = [2,3,4];
# x = [6,7,8,9,10];
# fn = x -> sum(a[idxs].*x[idxs]);
# fn(x)          # 74, initially.
# idxs = [1,2];  # Changing indices...
# fn(x)          # 20, changes value of function.
# # Changes in 'a' have a similar effect;
# a = [0,2,3,4,5];
# fn(x) # 14

# New JuliaLang example
a_mat = [[1,2,3,4,5],
         [2,3,4,5,6],
         [3,4,5,6,7]];
x = [6,7,8,9,10];
fns = [];
for i=1:3
    function fn(x, a = deepcopy(a_mat[i]))
        return sum(a.*x);
    end
    push!(fns, fn);
end
println([i(x) for i in fns]) # [130, 170,210]
a_mat[2][2] = 10;
println([i(x) for i in fns]) # [130, 219, 210]

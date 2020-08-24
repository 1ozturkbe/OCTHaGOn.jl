#=
test_src:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-06-16
This tests everything to do with the core of the OptimalConstraintTree code.
All the rest of the tests look at examples
=#

# Initialization tests
md = ModelData(c = [1,2,3]);

# Check infeasible bounds for Model Data
lbs = Dict(md.vks .=> [-2,1,3]);
ubs = Dict(md.vks .=> [Inf, 5, 6])
update_bounds!(md, lbs=lbs, ubs=ubs);
ubs = Dict(md.vks .=> [Inf, -1, 6])
@test_throws OCTException update_bounds!(md, ubs = ubs);

# Check sampling Model Data
@test_throws OCTException X = lh_sample(md, n_samples=100)

# # Check creation of JuMP.Model() from ModelData
jump_it!(md);
@test length(md.JuMP_model.linconstr) == 5;

# Test CBF imports
filename = string("data/cblib/shortfall_20_15.cbf.gz");
mof_model = CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
@test mof_obj ≈ -1.0792654303
#
# # Testing CBF import to ModelData
md = CBF_to_ModelData(filename);
md.name = "shortfall_20_15"
find_bounds!(md, all_bounds = true);
update_bounds!(md, lbs = Dict(vk => 0 for vk in md.vks));
update_bounds!(md, ubs = Dict(vk => 1 for vk in md.vks));

# Test sampling
n_samples = 500;
X = lh_sample(md, n_samples=n_samples);
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
sample_and_eval!(md, n_samples=500);
@test size(bbf.X) == (500,length(bbf.vks))
@test all(feasibility(md) .>= 0)

# Testing use KNN sampling and building trees
while any(feasibility(md) .<= 0.15)
    sample_and_eval!(md, n_samples=500);
end

learn_constraint!(md);

# Solving the model
status = globalsolve(md)
OCT_vars = JuMP.getvalue(md.JuMP_vars);
feasible, infeasible = evaluate_feasibility(md);

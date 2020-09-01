#=
test_src:
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
# TODO: add check for number of linear constraints in md.JuMP_Model.

# Test CBF imports
filename = string("data/cblib/shortfall_20_15.cbf.gz");
model = JuMP.read_from_file(filename);
set_optimizer(model, Gurobi.Optimizer)
optimize!(model);
moi_obj = JuMP.getobjectivevalue(model);
@test moi_obj ≈ -1.0792654303
#
# # Testing CBF import to ModelData
md = OptimalConstraintTree.CBF_to_ModelData(filename);
md.name = "shortfall_20_15"
find_bounds!(md, all_bounds = true);
update_bounds!(md, lbs = Dict(md.vks .=> 0.));
update_bounds!(md, ubs = Dict(md.vks .=> 1.));

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
OCT_vars = JuMP.getvalue.(md.vars);
feasible, infeasible = evaluate_feasibility(md);


# Doing CBF stuff
# model = JuMP.read_from_file(filename)
# cons_types = list_of_constraint_types(model)
#
# for (expr, type) in cons_types
#     cons = all_constraints(model, expr, type)
#     for consref in cons
#         one_constr = constraint_object(cons)
#     end
# end

#
# num_constraints(model, VariableRef, MOI.GreaterThan{Float64})
#
# num_constraints(model, VariableRef, MOI.ZeroOne)
#
# num_constraints(model, AffExpr, MOI.LessThan{Float64})
#
# constraint_object(con_ref::ConstraintRef)
#
#     F = MOI.get(model, MOI.ObjectiveFunctionType())
#
# MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
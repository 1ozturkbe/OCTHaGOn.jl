#=
cbf:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-09-02
=#

# Test CBF imports
#     gams01, netmod_dol2, netmod_kar1, netmod_kar2, du-opt, du-opt5, nvs03, ex1223, ex1223a, ex1223b, gbd
filename = string("data/cblib/shortfall_20_15.cbf.gz");
# Potential examples from MINLPLib
# gams01, netmod_dol2, netmod_kar1, netmod_kar2, du-opt, du-opt5, nvs03, ex1223, ex1223a, ex1223b, gbd
model = JuMP.read_from_file(filename);
set_optimizer(model, Gurobi.Optimizer)
optimize!(model);
moi_obj = JuMP.getobjectivevalue(model);
@test moi_obj ≈ -1.0792654303

# Testing CBF import to ModelData
md = OCTHaGOn.CBF_to_ModelData(filename);
md.name = "shortfall_20_15"
find_bounds!(md, all_bounds = true);
update_bounds!(md, lbs = Dict(md.vks .=> 0.));
update_bounds!(md, ubs = Dict(md.vks .=> 1.));

# Testing constraint import.
n_samples = 200;
bbl = md.bbls[1];
X = lh_sample(bbl, n_samples=n_samples);
Y = md.bbls[1](X);

# Testing sample_and_eval for combined LH and boundary sampling.
uniform_sample_and_eval!(md);
@test size(bbl.X) == (500,length(bbl.vks))
@test all(feasibility(md) .>= 0)

learn_constraint!(md);

# Solving the model
status = optimize!(md)
OCT_vars = JuMP.getvalue.(md.vars);
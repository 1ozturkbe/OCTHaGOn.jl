#=
test_sampling:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-07
=#

# Import a model
filename = string("../data/cblib.zib.de/shortfall_20_15.cbf.gz");
md = CBF_to_ModelData(filename);
find_bounds!(md)
update_bounds!(md, lbs = Dict(vk => 0 for vk in md.vks));
update_bounds!(md, ubs = Dict(vk => 1 for vk in md.vks));

# Pick one of the functions, and sample.
bbf = md.fns[1];
bbf.n_samples= 500;
sample_and_eval!(bbf); # initial is uniform!
println("Feasibility: ", bbf.feas_ratio)
sample_and_eval!(bbf); # then knn!
println("Feasibility: ", bbf.feas_ratio)
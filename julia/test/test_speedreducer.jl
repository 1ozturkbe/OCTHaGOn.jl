#=
test_speedreducer:
- Julia version: 
- Author: Berk
- Date: 2020-08-10
Speed reducer problem from [Li, 2017]
=#

using Test
using Combinatorics
using Gurobi
using JuMP
using LatinHypercubeSampling
using MosekTools
using MathOptInterface
using NearestNeighbors

include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;
global MOI = MathOptInterface
global PROJECT_ROOT = @__DIR__

function speed_reducer()
    c = zeros(8); c[end] = 1;
    md = OCT.ModelData(c = c);
    BBF = OCT.BlackBoxFunction;

    # Objective
    obj(x) = x[:x8] - (0.7854*x[:x1]*x[:x2]^2*(10. / 3. *x[:x3]^2 + 14.9334*x[:x3] - 43.0934) -
             1.508*x[:x1]*(x[:x6]^2 + x[:x7]^2) + 7.477*(x[:x6]^3 + x[:x7]^3) +
             0.7854*(x[:x4]*x[:x6]^2 + x[:x5]*x[:x7]^2))
    OCT.add_fn!(md, BBF(fn = obj, vks = md.vks, name = "obj"))
    OCT.update_bounds!(md, lbs = Dict(md.vks .=> [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5, 0]),
                           ubs = Dict(md.vks .=> [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5, 5000]))

    # Constraints
    OCT.add_fn!(md, BBF(name = "g1", fn = x -> -27 + x[:x1] * x[:x2]^2 * x[:x3], vks = [:x1, :x2, :x3]))

    OCT.add_fn!(md, BBF(name = "g2", fn = x -> -397.5 + x[:x1] * x[:x2]^2 * x[:x3]^2, vks = [:x1, :x2, :x3]))

    OCT.add_fn!(md, BBF(name = "g3", fn = x -> -1.93 + x[:x2] * x[:x6]^4 * x[:x3] / x[:x4]^3, vks = [:x2, :x3, :x4, :x6]))

    OCT.add_fn!(md, BBF(name = "g4", fn = x -> -1.93 + x[:x2] * x[:x7]^4 * x[:x3] / x[:x5]^3, vks = [:x2, :x3, :x5, :x7]))

    OCT.add_fn!(md, BBF(name = "g5", fn = x -> 1100 - (((745*x[:x4]/(x[:x2]*x[:x3]))^2) +
                                            16.91*10^6)^0.5/(0.1*x[:x6]^3),
                                            vks = [:x2, :x3, :x4, :x6]))

    OCT.add_fn!(md, BBF(name = "g6", fn = x -> 850 - (((745*x[:x5]/(x[:x2]*x[:x3]))^2) +
                                            157.5*10^6)^0.5/(0.1*x[:x7]^3),
                                            vks = [:x2, :x3, :x5, :x7]))

    OCT.add_fn!(md, BBF(name = "g7", fn = x -> 40 - x[:x2]*x[:x3], vks = [:x2, :x3]))

    OCT.add_fn!(md, BBF(name = "g8", fn = x -> x[:x1]/x[:x2] - 5, vks = [:x1, :x2]))
    OCT.add_fn!(md, BBF(name = "g9", fn = x -> 12 - x[:x1]/x[:x2], vks = [:x1, :x2]))
    OCT.add_fn!(md, BBF(name = "g10", fn = x -> x[:x4] - 1.5*x[:x6] - 1.9, vks = [:x4, :x6]))
    # It turns out constraint g11 is actually completely infeasible... how could they publish these results?
#     OCT.add_fn!(md, BBF(name = "g11", fn = x -> x[:x5] - 1.5*x[:x7] - 1.9, vks = [:x5, :x7]))
    return md
end

md = speed_reducer()

# Initial sampling
OCT.sample_and_eval!(md, n_samples=200, iterations=1)
println("Constraint feasibilities: ", [fn.feas_ratio for fn in md.fns])
feas_names, infeas_names = OCT.fns_by_feasibility(md)

for name in infeas_names
    bbf = md(name);
    OCT.build_knn_tree(bbf);
    idxs, dists = OCT.find_knn(bbf, k=10);
    class_dict = OCT.classify_patches(bbf, idxs);
end

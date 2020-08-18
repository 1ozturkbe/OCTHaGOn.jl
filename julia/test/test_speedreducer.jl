#=
test_speedreducer:
- Julia version: 
- Author: Berk
- Date: 2020-08-10
Speed reducer problem from [Li, 2017]
=#

function speed_reducer()
    c = zeros(8); c[end] = 1;
    md = ModelData(c = c);
    BBF = BlackBoxFunction;

    # Objective
    obj(x) = x[:x8] - (0.7854*x[:x1]*x[:x2]^2*(10. / 3. *x[:x3]^2 + 14.9334*x[:x3] - 43.0934) -
             1.508*x[:x1]*(x[:x6]^2 + x[:x7]^2) + 7.477*(x[:x6]^3 + x[:x7]^3) +
             0.7854*(x[:x4]*x[:x6]^2 + x[:x5]*x[:x7]^2))
    add_fn!(md, BBF(fn = obj, vks = md.vks, name = "obj"))
    update_bounds!(md, lbs = Dict(md.vks .=> [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5, 0]),
                           ubs = Dict(md.vks .=> [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5, 5000]))

    # Constraints
    add_fn!(md, BBF(name = "g1", fn = x -> -27 + x[:x1] * x[:x2]^2 * x[:x3], vks = [:x1, :x2, :x3]))

    add_fn!(md, BBF(name = "g2", fn = x -> -397.5 + x[:x1] * x[:x2]^2 * x[:x3]^2, vks = [:x1, :x2, :x3]))

    add_fn!(md, BBF(name = "g3", fn = x -> -1.93 + x[:x2] * x[:x6]^4 * x[:x3] / x[:x4]^3, vks = [:x2, :x3, :x4, :x6]))

    add_fn!(md, BBF(name = "g4", fn = x -> -1.93 + x[:x2] * x[:x7]^4 * x[:x3] / x[:x5]^3, vks = [:x2, :x3, :x5, :x7]))

    add_fn!(md, BBF(name = "g5", fn = x -> 1100 - (((745*x[:x4]/(x[:x2]*x[:x3]))^2) +
                                            16.91*10^6)^0.5/(0.1*x[:x6]^3),
                                            vks = [:x2, :x3, :x4, :x6]))

    add_fn!(md, BBF(name = "g6", fn = x -> 850 - (((745*x[:x5]/(x[:x2]*x[:x3]))^2) +
                                            157.5*10^6)^0.5/(0.1*x[:x7]^3),
                                            vks = [:x2, :x3, :x5, :x7]))

    add_fn!(md, BBF(name = "g7", fn = x -> 40 - x[:x2]*x[:x3], vks = [:x2, :x3]))

    add_fn!(md, BBF(name = "g8", fn = x -> x[:x1]/x[:x2] - 5, vks = [:x1, :x2]))
    add_fn!(md, BBF(name = "g9", fn = x -> 12 - x[:x1]/x[:x2], vks = [:x1, :x2]))
    add_fn!(md, BBF(name = "g10", fn = x -> x[:x4] - 1.5*x[:x6] - 1.9, vks = [:x4, :x6]))
    # It turns out constraint g11 is actually completely infeasible... how could they publish these results?
#     add_fn!(md, BBF(name = "g11", fn = x -> x[:x5] - 1.5*x[:x7] - 1.9, vks = [:x5, :x7]))
    return md
end

md = speed_reducer()

# Initial sampling (boundary and interior)
sample_and_eval!(md, n_samples=200, iterations=1)
println("Constraint feasibilities: ", feasibility(md))

# See if KNN sampling makes a difference for feasibility!
sample_and_eval!(md, n_samples=200, iterations=1)
println("Constraint feasibilities: ", feasibility(md))

learn_constraint!(md)

globalsolve(md)
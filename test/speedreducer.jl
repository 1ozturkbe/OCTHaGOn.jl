#=
test_speedreducer:
- Julia version: 
- Author: Berk
- Date: 2020-08-10
Speed reducer problem from [Li, 2017]
=#

function speed_reducer()
    m = JuMP.Model(Gurobi.Optimizer)
    @variable(m, x[1:8])
    @objective(m, Min, x[8])
    BBF = BlackBoxFunction;
    lbs = Dict(x .=> [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5, 0])
    ubs = Dict(x .=> [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5, 5000])
    gm = GlobalModel(model = m, name = "speed reducer")
    bound!(gm, Dict(var => [lbs[var], ubs[var]] for var in x))
    all_vars = all_variables(gm)

    # Objective
    add_nonlinear_constraint(gm, @NLexpression(gm.model, x[8] - (0.7854*x[1]*x[2]^2*(10. / 3. *x[3]^2 + 14.9334*x[3] - 43.0934) -
             1.508*x[1]*(x[6]^2 + x[7]^2) + 7.477*(x[6]^3 + x[7]^3) +
             0.7854*(x[4]*x[6]^2 + x[5]*x[7]^2))), vars = all_vars)


    # Constraints
    add_nonlinear_constraint(gm, @NLexpression(gm.model, g1, -27 + x[1] * x[2]^2 * x[3]), vars = x[1:3])

    add_nonlinear_constraint(gm, @NLexpression(gm.model, g2, -397.5 + x[1] * x[2]^2 * x[3]^2), vars = x[1:3])

    add_nonlinear_constraint(gm, @NLexpression(gm.model, g3, -1.93 + x[2] * x[6]^4 * x[3] / x[4]^3),  vars = [x[2], x[3], x[4], x[6]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g4, -1.93 + x[2] * x[7]^4 * x[3] / x[5]^3), vars = [x[2], x[3], x[5], x[7]]))

    add_nonlinear_constraintgm, @NLexpression(gm.model, g5, 1100 - (((745*x[4]/(x[2]*x[3]))^2) +
                                            16.91*10^6)^0.5/(0.1*x[6]^3)), vars = [x[2], x[3], x[4], x[6]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g6, 850 - (((745*x[5]/(x[2]*x[3]))^2) +
                                            157.5*10^6)^0.5/(0.1*x[7]^3)), vars = [x[2], x[3], x[5], x[7]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g7, 40 - x[2]*x[3]), vars = [x[2], x[3]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g8, x[1]/x[2] - 5), vars = [x[1], x[2]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g9, 12 - x[1]/x[2]), vars = [x[1], x[2]])

    add_nonlinear_constraintgm, @NLexpression(gm.model, g10, x[4] - 1.5*x[6] - 1.9), vars = [x[4], x[6]])

    # It turns out constraint g11 is actually completely infeasible... how could they publish these results?
#     add_fn!(md, BBF(name = "g11", fn = x -> x[5] - 1.5*x[7] - 1.9, vks = [:x5, :x7]))
    return gm
end

gm = speed_reducer()
n_samples = 100

# Initial sampling (boundary and interior)
sample_and_eval!(md, n_samples=n_samples)
println("Constraint feasibilities: ", feasibility(md))

# See if KNN sampling makes a difference for feasibility!
sample_and_eval!(md)
println("Constraint feasibilities: ", feasibility(md))

feas, infeas = fns_by_feasibility(md)
for idx in infeas
    sample_and_eval!(md(idx))
end

learn_constraint!(md)

globalsolve(md)

x_vals = getvalue.(md.vars);
feasible, infeasible = evaluate_feasibility(md);

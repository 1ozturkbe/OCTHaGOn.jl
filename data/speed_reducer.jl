#=
Ray, T. (2003). Golinski’s speed reducer problem revisited. AIAA Journal, 41(3), 556–558. https://doi.org/10.2514/2.1984
=#

function speed_reducer(solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, x[1:8])
    @objective(m, Min, x[8])
    gm = GlobalModel(model = m, name = "speed_reducer")
    lbs = Dict(x .=> [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5, 0])
    ubs = Dict(x .=> [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5, 5000])
    JuMP.set_integer(x[3])
    bound!(gm, Dict(var => [lbs[var], ubs[var]] for var in gm.vars))

    # Objective
    # add_nonlinear_constraint(gm, :(x -> x[8] - (0.7854*x[1]*x[2]^2*(10. / 3. *x[3]^2 + 14.9334*x[3] - 43.0934) -
    #          1.5079*x[1]*(x[6]^2 + x[7]^2) + 7.477*(x[6]^3 + x[7]^3) +
    #          0.7854*(x[4]*x[6]^2 + x[5]*x[7]^2))), vars = x[1:8])
    add_nonlinear_constraint(gm, :(x -> 0.7854*x[1]*x[2]^2*(3.3333 *x[3]^2 + 14.9334*x[3] - 43.0934) -
    1.5079*x[1]*(x[6]^2 + x[7]^2) + 7.477*(x[6]^3 + x[7]^3) +
    0.7854*(x[4]*x[6]^2 + x[5]*x[7]^2)), vars = x[1:7], dependent_var = x[8])

    # Constraints
    add_nonlinear_constraint(gm, :(x -> -27 + x[1] * x[2]^2 * x[3]), vars = x[1:3])

    add_nonlinear_constraint(gm, :(x -> -397.5 + x[1] * x[2]^2 * x[3]^2), vars = x[1:3])

    add_nonlinear_constraint(gm, :(x -> -1.93 + x[2] * x[6]^4 * x[3] / x[4]^3),  vars = [x[2], x[3], x[4], x[6]])

    add_nonlinear_constraint(gm, :(x -> -1.93 + x[2] * x[7]^4 * x[3] / x[5]^3), vars = [x[2], x[3], x[5], x[7]])

    add_nonlinear_constraint(gm, :(x -> 110.0*x[6]^3 - (((745*x[4]/(x[2]*x[3]))^2) +
    16.9*10^6)^0.5), vars = [x[2], x[3], x[4], x[6]])

    add_nonlinear_constraint(gm, :(x -> 85.0*x[7]^3 - (((745*x[5]/(x[2]*x[3]))^2) +
                                            157.5*10^6)^0.5), vars = [x[2], x[3], x[5], x[7]])

    add_nonlinear_constraint(gm, :(x -> 40 - x[2]*x[3]), vars = [x[2], x[3]])

    add_nonlinear_or_compatible(gm, :(x -> x[1] - 5*x[2]), vars = [x[1], x[2]])

    add_nonlinear_or_compatible(gm, :(x -> 12*x[2] - x[1]), vars = [x[1], x[2]])

    add_nonlinear_or_compatible(gm, :(x -> x[4] - 1.5*x[6] - 1.9), vars = [x[4], x[6]])

    add_nonlinear_or_compatible(gm, :(x -> x[5] - 1.1*x[7] - 1.9), vars = [x[5], x[7]])
    
    set_param(gm, :ignore_accuracy, true)
    
    return gm
end
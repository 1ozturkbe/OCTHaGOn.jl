# // process synthes1
# // mixed-integer nonlinear program
#
# // Source: Test problem 1 (Synthesis of processing system) in
# // M. Duran & I.E. Grossmann,
# // "An outer approximation algorithm for a class of mixed integer nonlinear
# //  programs", Mathematical Programming 36, pp. 307-339, 1986.
#
# // Number of variables:   6 (3 binary variables)
# // Number of constraints: 6
# // Objective nonlinear
# // Nonlinear constraints

function minlp(oct::Bool = false)
    m = JuMP.Model()
    @variable(m, x[1:6] >= 0)
    JuMP.set_binary.(x[4:6])
    JuMP.set_upper_bound(x[1], 2)
    JuMP.set_upper_bound(x[2], 2)
    JuMP.set_upper_bound(x[3], 1)
    JuMP.set_upper_bound(x[4], 1)
    JuMP.set_upper_bound(x[5], 1)
    JuMP.set_upper_bound(x[6], 1)
    
    @constraint(m, c3, x[2] - x[1] <= 0)
    @constraint(m, c4, x[2] - 2*x[4] <= 0)
    @constraint(m, c5, x[1] - x[2] - 2*x[5] <= 0)
    @constraint(m, c6, x[4] + x[5] <= 1)

    if !oct
        @NLconstraint(m, c1, 0.8*log(x[2] + 1) + 0.96*log(x[1] - x[2] + 1) - 0.8*x[3] >= 0)
        @NLconstraint(m, c2, log(x[2] + 1) + 1.2*log(x[1] - x[2] + 1) - x[3] - 2*x[6] >= -2)
        @NLobjective(m, Min, 5*x[4] + 6*x[5] + 8*x[6] + 10*x[1] - 7*x[3] - 18*log(x[2] + 1) -
                             19.2*log(x[1] - x[2] + 1) + 10)
        return m
    else
        @variable(m, obj)
        @objective(m, Min, obj)
        gm = GlobalModel(model = m, name = "minlp")
        add_nonlinear_constraint(gm, :(x -> 0.8*log(x[2] + 1) + 0.96*log(x[1] - x[2] + 1) - 0.8*x[3]),
                                 name = "c1")
        add_nonlinear_constraint(gm, :(x -> log(x[2] + 1) + 1.2*log(x[1] - x[2] + 1) - x[3] - 2*x[6] + 2),
                                 name = "c2")
        add_nonlinear_constraint(gm, :(x -> 5*x[4] + 6*x[5] + 8*x[6] + 10*x[1] - 7*x[3] - 18*log(x[2] + 1) -
                             19.2*log(x[1] - x[2] + 1) + 10),
                                 name = "objective", dependent_var = obj)
        return gm
    end
end
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

function minlp(gm::Bool = false, eps = 1e-10)
    m = JuMP.Model()
    @variable(m, 0 <= y[1:3] <= 1, Int)
    @variable(m, x[1:3] >= eps)
    JuMP.set_upper_bound(x[1], 2)
    JuMP.set_upper_bound(x[2], 2)
    JuMP.set_upper_bound(x[3], 1)
    
    @constraint(m, c3, x[2] - x[1] <= 0)
    @constraint(m, c4, x[2] - 2*y[1] <= 0)
    @constraint(m, c5, x[1] - x[2] - 2*y[2] <= 0)
    @constraint(m, c6, y[1] + y[2] <= 1)

    if !gm
        @NLconstraint(m, c1, 0.8*log(x[2] + 1) + 0.96*log(x[1] - x[2] + 1) - 0.8*x[3] >= 0)
        @NLconstraint(m, c2, log(x[2] + 1) + 1.2*log(x[1] - x[2] + 1) - x[3] - 2*y[3] >= -2)
        @NLobjective(m, Min, 5*y[1] + 6*y[2] + 8*y[3] + 10*x[1] - 7*x[3] - 18*log(x[2] + 1) -
                             19.2*log(x[1] - x[2] + 1) + 10)
        set_optimizer(m, BARON_SILENT)
    else
        @variable(m, obj)
        @objective(m, Min, obj)
        gm = GlobalModel(model = m, name = "minlp")
        set_optimizer(gm, GUROBI_SILENT)
        add_nonlinear_constraint(gm, :(x -> 0.8*log(x[2] + 1) + 0.96*log(x[1] - x[2] + 1) - 0.8*x[3]),
                                 name = "c1")
        add_nonlinear_constraint(gm, :((x,y) -> log(x[2] + 1) + 1.2*log(x[1] - x[2] + 1) - x[3] - 2*y[3] + 2),
                                 name = "c2")
        add_nonlinear_constraint(gm, :((x, y, obj) -> obj - (5*y[1] + 6*y[2] + 8*y[3] + 10*x[1] - 7*x[3] - 18*log(x[2] + 1) -
                             19.2*log(x[1] - x[2] + 1) + 10)),
                                 name = "objective")
        return gm
    end
    return m
end
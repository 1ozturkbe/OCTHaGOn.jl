# solution is x1=6, x2=2/3; objective = -6.666667
function nlp2(gm::Bool = false)
    m = JuMP.Model()
    @variable(m, 0 <= x[1:2])
    for i = 1:2
        JuMP.set_lower_bound(x[i], [6, 4][i])
    end
    @objective(m, Min, -x[1] - x[2])
    if !gm
        @NLconstraint(m, e1, x1*x2 <= 4)
        set_optimizer(m, BARON_SILENT)
    else
        gm = GlobalModel(model = m, name = "nlp1")
        set_optimizer(gm, CPLEX_SILENT)
        add_nonlinear_constraint(gm, :(x ->  4 - x[1]*x[2]), name = "e1")
    end
    return m
end
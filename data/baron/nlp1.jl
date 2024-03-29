# solution is x1=6, x2=2/3; objective = -6.666667
function nlp1(oct::Bool = false)
    m = JuMP.Model()
    @variable(m, 0 <= x[1:2])
    for i = 1:2
        JuMP.set_upper_bound(x[i], [6, 4][i])
    end
    @objective(m, Min, -x[1] - x[2])
    if !oct
        @NLconstraint(m, e1, x[1]*x[2] <= 4)
        return m
    else
        gm = GlobalModel(model = m, name = "nlp1")
        add_nonlinear_constraint(gm, :(x ->  4 - x[1]*x[2]), name = "e1")
        return gm
    end
end
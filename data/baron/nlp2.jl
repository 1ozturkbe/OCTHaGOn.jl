# (Stoecker, 1971; modified)
# solution is: x1=6.2934300, x[2]=3.8218391, x[3]=201.1593341
# objective = 201.1593341  

function nlp2(gm::Bool = false)
    m = JuMP.Model()
    @variable(m, 0 <= x[1:3])
    for i = 1:3
        JuMP.set_upper_bound(x[i], [9.422, 5.9023, 267.417085245][i])
    end
    @objective(m, Min, x[3])
    if !gm
        @NLconstraint(m, e1, 250 + 30*x[1] - 6*x[1]^2 - x[3] == 0)
        @NLconstraint(m, e2, 300 + 20*x[2] - 12*x[2]^2 - x[3] == 0)
        @NLconstraint(m, e3, 150 + 0.5*(x[1]+x[2])^2 - x[3] == 0)
        set_optimizer(m, BARON_SILENT)
        return m
    else
        gm = GlobalModel(model = m, name = "nlp2")
        set_optimizer(gm, CPLEX_SILENT)
        add_nonlinear_constraint(gm, :(x -> 250 + 30*x[1] - 6*x[1]^2 - x[3]), name = "e1", equality = true)
        add_nonlinear_constraint(gm, :(x -> 300 + 20*x[2] - 12*x[2]^2 - x[3]), name = "e2", equality = true)
        add_nonlinear_constraint(gm, :(x ->  150 + 0.5*(x[1]+x[2])^2 - x[3]), name = "e3", equality = true)
        return gm
    end
end

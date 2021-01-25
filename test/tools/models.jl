""" Test JuMP model with a variety of variable bounds and sizes. """
function test_model()
    model = Model()
    @variables(model, begin
        -5 <= x[1:5] <= 5
        -4 <= y[1:3] <= 1
        -30 <= z
        -2 <= a[1:2, 1:3]
    end)
    JuMP.set_upper_bound(a[2,2], 3)
    JuMP.set_upper_bound(a[1,1], 3)
    return model, x, y, z, a
end

""" Basic qp for testing. """
function test_qp(solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, 2 >= x[1:2] >= 0) 
    @objective(m, Min, 3*x[1]^2 + x[2]^2 + 2*x[1]*x[2] + x[1] + 6*x[2] + 2)
    @constraint(m, 2*x[1] + 3*x[2] >= 4)
    return m
end

""" Basic gqp, same as above. """
function test_gqp(solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, 2 >= x[1:2] >= 0) 
    @constraint(m, 2*x[1] + 3*x[2] >= 4)
    @variable(m, obj)
    @objective(m, Min, obj)
    gm = GlobalModel(model = m)
    add_nonlinear_constraint(gm, :(x -> 3*x[1]^2 + x[2]^2 + 2*x[1]*x[2] + x[1] + 6*x[2] + 2), dependent_var = obj)
    return gm
end
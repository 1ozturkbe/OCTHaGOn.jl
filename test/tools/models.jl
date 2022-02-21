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
function test_qp(solver = OCTHaGOn.SOLVER_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, 2 >= x[1:2] >= 0) 
    @objective(m, Min, 3*x[1]^2 + x[2]^2 + 2*x[1]*x[2] + x[1] + 6*x[2] + 2)
    @constraint(m, 2*x[1] + 3*x[2] >= 4)
    return m
end

""" Basic gqp, same as above. """
function test_gqp(solver = OCTHaGOn.SOLVER_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, 2 >= x[1:2] >= 0) 
    @constraint(m, 2*x[1] + 3*x[2] >= 4)
    @variable(m, obj)
    @objective(m, Min, obj)
    gm = GlobalModel(model = m)
    add_nonlinear_constraint(gm, :(x -> 3*x[1]^2 + x[2]^2 + 2*x[1]*x[2] + x[1] + 6*x[2] + 2), dependent_var = obj)
    return gm
end

"""
    random_qp(dims::Int64, nconstrs::Int64, sparsity=dims, solver = OCTHaGOn.SOLVER_SILENT)

Generates a random quadratic program with the specified parameters. 
"""
function random_qp(dims::Int64, nconstrs::Int64, sparsity=dims, solver = OCTHaGOn.SOLVER_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, -1 <= x[1:dims] <= 1)
    m[:sparse_vars] = randperm(dims)[1:sparsity]
    m[:A] = rand(dims, sparsity).*2 .- 1
    m[:b] = rand(dims)
    m[:C] = rand(nconstrs, dims).*2 .- 1
    m[:d] = rand(nconstrs)
    @constraint(m, m[:C]*x .>= m[:d])
    @objective(m, Min, sum((m[:A]*x[m[:sparse_vars]] - m[:b]).^2))
    return m
end

""" Turns random_qp into a GlobalModel. """
function gmify_random_qp(m::JuMP.Model)
    @variable(m, obj)
    @objective(m, Min, obj)
    gm = GlobalModel(model = m)
    add_nonlinear_constraint(gm, :(x -> sum(($(m[:A])*x[$(m[:sparse_vars])] - $(m[:b])).^2)), vars = m[:x][m[:sparse_vars]], 
                                    dependent_var = obj, name = "qp") 
    return gm
end
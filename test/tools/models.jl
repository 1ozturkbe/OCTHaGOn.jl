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
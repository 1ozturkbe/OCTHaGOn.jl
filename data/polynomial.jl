"""
Polynomial optimization example from [Bertsimas, Nohadani, 2010]
"""
function poly1()
    m = JuMP.Model()
    @variable(m, -1 <= x <= 4)
    @variable(m, -1 <= y <= 4)
    @variable(m, obj)
    @objective(m, Min, obj)
    gm = GlobalModel(model = m)
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)
    set_param(gm, :abstol, 1e-3)

    add_nonlinear_constraint(gm, :((x,y) -> 2*x^6 - 12.2*x^5 + 21.2*x^4 + 6.2*x - 6.4*x^3 - 4.7*x^2 + 
    y^6 - 11*y^5 + 43.3*y^4 - 10*y - 74.8*y^3 + 56.9*y^2 - 4.1*x*y - 0.1*y^2*x^2 + 0.4*y^2*x + 0.4*x^2*y), 
    name = "objective", dependent_var = obj)
    add_nonlinear_constraint(gm, :((x,y) -> 10.125 - (x-1.5)^4 - (y-1.5)^4), name = "h1")
    add_nonlinear_constraint(gm, :((x,y) -> (2.5 - x)^3 + (y+1.5)^3 - 15.75), name = "h2")
    return gm 
end 
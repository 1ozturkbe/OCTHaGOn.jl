""" Predator prey model with logistic function from http://www.math.lsa.umich.edu/~rauch/256/F2Lab5.pdf """
function foxes_and_rabbits(oct::Bool = true)
    m = Model(SOLVER_SILENT)
    t = 20
    r = 0.2
    x1 = 0.6
    y1 = 0.5
    @variable(m, x[1:t] >= 0.01) # Note: Ipopt solution does not converge with an upper bound!!
    @variable(m, dx[1:t-1])
    @variable(m, y[1:t] >= 0.01)
    @variable(m, dy[1:t-1])
    @constraint(m, x[1] == x1)
    @constraint(m, y[1] == y1)
    @constraint(m, [i=2:t], x[i] == x[i-1] + dx[i-1])
    @constraint(m, [i=2:t], y[i] == y[i-1] + dy[i-1])
    
    # GlobalModel representation
    set_upper_bound.(x, 1)
    set_upper_bound.(dx, 1)
    set_lower_bound.(dx, -1)
    set_upper_bound.(y, 1)
    set_upper_bound.(dy, 1)
    set_lower_bound.(dy, -1)

    if !oct
        @NLconstraint(m, [i=1:t-1], dx[i] == x[i]*(1-x[i]) - x[i]*y[i]/(x[i]+1/5))
        @NLconstraint(m, [i=1:t-1], dy[i] == r*y[i]*(1-y[i]/x[i]))
        return m
    else
        gm = GlobalModel(model = m, name = "foxes_rabbits")
        add_nonlinear_constraint(gm, :((x, y, dx) -> dx[1] - (x[1]*(1-x[1]) -x[1]*y[1]/(x[1]+$(r)))), 
                                vars = [x[1], y[1], dx[1]], equality=true)
        add_nonlinear_constraint(gm, :((x, y, dy) -> dy[1] - $(r)*y[1]*(1-y[1]/x[1])), 
                                vars = [x[1], y[1], dy[1]], equality=true)
        for i = 2:t-1
            add_linked_constraint(gm, gm.bbls[1], [x[i], y[i], dx[i]])
            add_linked_constraint(gm, gm.bbls[2], [x[i], y[i], dy[i]])
        end
        return gm
    end
end
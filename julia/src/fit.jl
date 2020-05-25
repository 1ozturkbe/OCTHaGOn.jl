using Gurobi
using JuMP
# Defining regression function, with rho and pnorm
function regress(Y, X, rho, p, M=2.)
    m = Model(solver=GurobiSolver())
    n_points = size(X, 1); # number of data points
    n_traits = size(X, 2); # number of traits
    @variable(m, b[i=1:n_traits])
    @variable(m, b0)
    @variable(m, t) # fitting error
    @variable(m, s) # sparsity penalty
    @constraint(m, sum((Y - X*b - b0).^2) <= t) # The fit cost
    if p == 0.
        @variable(m, z[1:n_traits], Bin)
        @constraint(m, scon, rho*sum(z[:]) <= s)        
        @constraint(m, b[:] .<= M*z[:])
        @constraint(m, -b[:] .<= M*z[:])
    elseif p == 1.
        @variable(m, pb[1:n_traits])
        @constraint(m, pb[:] .>= b[:])
        @constraint(m, pb[:] .>= -b[:])
        @constraint(m, scon, rho*sum(pb) <= s)
    elseif p == 2.
        @constraint(m, scon, rho*sum(b.^2) <= s)
    else
        print("Error: this norm is not supported.")
    end
    @objective(m, Min, t + s);
    status = solve(m)
    return getvalue(b), getvalue(b0), getvalue(t), getvalue(s)
end
using JuMP
using Gurobi

function logistic_conjugate(y,alpha)
    return (1+y'*alpha)*log(1+y'*alpha) - y'*alpha*log(-y'*alpha)
end
function logistic_regress(X, Y, k, gamma, M=10)
    n, p = size(X)
    m = Model(solver=GurobiSolver())
    @variable(m, eta)
    @objective(m, Min, eta)
    # Binary variables
    @variable(m, s[1:p], Bin)
    @constraint(m, sum(s[:]) <= k)
    # Regression variables with big M
    @variable(m, beta[1:p])
    # Dual variables
    @variable(m, alpha[1:n])
    @constraint(m, sum(alpha) == 0)
    s_0 = zeros(p,1);
    s_0[1:k] .= 1.
    solve(m)
    # Adding constraint generator
    function subgradient(cb)
        s_i = getvalue(s)
        alpha_i = getvalue(alpha)
        print("In callback f'n, s_i=$s_i")
        grad = [-gamma/2*alpha_i'*(X[j]*X[j]')*alpha_i for j=1:p]
        @lazyconstraint(cb, eta .>= logistic_conjugate(s_i) + grad'*(s-s_i))
    end
    addlazycallback(m, inner_dual)
    solve(m)
    return getvalue(z)
end
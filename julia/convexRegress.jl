using Gurobi
using Mosek
using JuMP
using Random
function convexRegress(Y, X, gamma, cut_fraction=1.0, epsilon = 1e-5)
    n = size(X,1); p = size(X,2);
    m = Model(solver=GurobiSolver())
    @variable(m, theta[1:n])
    @variable(m, ksi[1:n, 1:p])
    @objective(m, Min, 0.5*sum((Y-theta).^2) + 1/(2*gamma)*sum([sum(ksi[i,:].^2) for i=1:n-1]))
    # Creating constraints for ordered thetas
    for i=1:size(ksi,1)-1
        @constraint(m, theta[i] + sum(ksi[i,:].*(X[i+1,:] - X[i,:])) <= theta[i+1])
        @constraint(m, theta[i+1] + sum(ksi[i+1,:].*(X[i,:] - X[i+1,:])) <= theta[i])
    end
    @constraint(m, theta[n] + sum(ksi[n,:].*(X[1,:] - X[n,:])) <= theta[1])
    @constraint(m, theta[1] + sum(ksi[1,:].*(X[n,:] - X[1,:])) <= theta[n])
    # Adding constraint generator, since callbacks don't work
    violation = epsilon;
    cut_count = 1;
    while violation >= epsilon && cut_count <= size(X,1)^2/2*cut_fraction
        solve(m)
        ksis = getvalue(ksi); thetas = getvalue(theta);
        for i=1:n
            violation = 0;
            cut_index = 0;
            for k=1:n
                if (thetas[i] - thetas[k] + sum(ksis[i,:].*(X[k,:] - X[i,:]))) >= violation && (i != k)
                    violation = thetas[i] - thetas[k] + sum(ksis[i,:].*(X[k,:] - X[i,:]));
                    cut_index = k;
                end
            end
            if violation > epsilon
                @constraint(m, thetas[i] + sum(ksis[i,:].*(X[cut_index,:] - X[i,:])) <= thetas[cut_index])
                cut_count += 1;
#                 println("Cut ", cut_count, " added: ", [i,cut_index])
            end
        end
    end
    println("Cuts added: ", cut_count)
    return getvalue(theta), getvalue(ksi)
end

# function convexRegressDual(Y, X, gamma, k)
#     n = size(X,1); p = size(X,2);
#     m = Model(solver=GurobiSolver())
#     @variable(m, z[1:p], Bin)
#     @variable(m, nu)
#     @variable(m, mu[1:n, 1:n])
#     # Objective
#     @objective(m, Max, nu)
#     # Subgradient and objective f'ns
#     function gz(Y, X, mu, z)
#         n = size(X,1); p = size(X,2);
#         return -0.5*sum([(y[i] + sum(mu[:,i]) - sum(mu[i,:])).^2 for i=1:n]) - 
#                 0.5*gamma*sum(sum())
#     end
#     function dgz(Y, X, mu, z)
#         n = size(X,1); p = size(X,2);
#         dg = zeros(p);
#         for i = 1:p
#             dg[i] = -0.5/gamma*sum(sum(mu[i,j]*(X[i,p] - X[j,p]) for j=1:n)^2 for i=1:n)
#         end
#         return dg
#     end
    
#     # Initialize cuts
#     z0 = zeros(p)
#     z0[1:k] .= 1;
        
#     return getvalue(theta), getvalue(ksi)
# end
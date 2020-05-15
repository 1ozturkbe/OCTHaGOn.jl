using JuGP
using JuMP
using Gurobi

function woodyGP(solver=GurobiSolver())
    m = GPModel(solver=solver)
    
    # Fixed and uncertain parameters
    @variable(m, k == 1.17)
    @variable(m, e == 0.92)
    @variable(m, mu == 1.775e-5)
    @variable(m, rho == 1.23)
    @variable(m, tau == 0.12)
    @variable(m, N_ult == 3.3)
    @variable(m, V_min == 25)
    @variable(m, C_Lmax == 1.6)
    @variable(m, S_wetratio == 2.075)
    @variable(m, W_W_coeff1 == 12e-5)
    @variable(m, W_W_coeff2 == 60)
    @variable(m, CDA0 == 0.035)
    @variable(m, W_0 == 6250)

    # Free variables
    @variable(m, A)
    @variable(m, S)
    @variable(m, D)
    @variable(m, C_D)
    @variable(m, C_D_fuse)
    @variable(m, C_D_ind)
    @variable(m, C_D_wpar)
    @variable(m, C_L)
    @variable(m, C_f)
    @variable(m, Re)
    @variable(m, W)
    @variable(m, W_w)
    @variable(m, W_w_strc)
    @variable(m, W_w_surf)
    @variable(m, V)
    
    #Objective
    @objective(m, Min, D)
    
    # Drag Model    
#     @NLconstraint(m, C_D_fuse == CDA0 / S)
#     @NLconstraint(m, C_D_wpar == k * C_f * S_wetratio)
#     @NLconstraint(m, C_D_ind == C_L ^ 2 / (pi * A * e))
    @NLconstraint(m, C_D >= C_D_ind + C_D_fuse + C_D_wpar)

    # Wing Weight Model
#     @NLconstraint(m, W_w_strc >= W_W_coeff1 * (N_ult * A ^ 1.5 * (W_0 * W * S) ^ 0.5) / tau)
#     @NLconstraint(m, W_w_surf >= W_W_coeff2 * S)
#     @NLconstraint(m, W_w >= W_w_surf + W_w_strc)

#     # and the rest of the models
#     @NLconstraint(m, D >= 0.5 * rho * S * C_D * V ^ 2)
#     @NLconstraint(m, Re <= (rho / mu) * V * (S / A) ^ 0.5)
#     @NLconstraint(m, C_f >= 0.074 / Re ^ 0.2)
#     @NLconstraint(m, W <= 0.5 * rho * S * C_L * V ^ 2)
#     @NLconstraint(m, W <= 0.5 * rho * S * C_Lmax * V_min ^ 2)
#     @NLconstraint(m, W >= W_0 + W_w)
    
    return m
end

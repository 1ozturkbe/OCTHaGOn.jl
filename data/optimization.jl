
using JuMP
using PiecewiseLinearOpt
using JSON

# Define data
include("data.jl")
@assert length(h_ρi) == length(ρi) == length(Hi)
@assert issorted(h_ρi)
@assert issorted(ρi, rev=true)
@assert issorted(Hi)
# Define global model in log-space
function global_model(solver, n_pts::Int, X_R, exp_Lt_min, B=8)
    X_R_original = X_R
    X_R = log(X_R)
    B = log(B)
    m = Model(solver=solver)

    # log variables
    @variables m begin
        # Subsystem specific parameters
        D_p # payload optical aperture diameter
        X_r # payload resolution
        G_T # transmitter antenna diameter
        A # solar panel surface area

        # catalog
        ρ_A
        η_A

        # Mass related parameters
        #X_r
        m_t >= m_min # total mass
        m_b # battery mass
        m_A # solar panel mass
        m_p # payload mass
        m_T # transmitter mass
        m_S # structural mass
        m_P # propulsion mass

        # Power related parameters
        P_t # total power
        P_T # transmitter power
        E_b # battery energy

        # Orbit related parameters
        h_min <= h <= h_max # altitude
        a # semi-major axis
        T # orbit period
        r # worst case communications distance
        d_min <= d <= d_max # daylight fraction of orbit
        e_min <= e <= e_max # eclipse fraction of orbit
        g_min <= g <= g_max # ground station viewing fraction of orbit

        # Lifetime (years)
        exp_Ln_min <= exp_Ln <= exp_Ln_max # without propulsion (orbit decay)
        exp_Lp_min <= exp_Lp <= exp_Lp_max # with propulsion
        ρ # atmospheric density
        H # atmosphere scale height
        h_ρi[1] <= h_ρ <= h_ρi[end]

        # Disjunctive
        T_g
        log(1e-2) <= m_P2 <= log(10) # mass for reaction wheel
        log(1e-2) <= m_M <= log(10) # mass for magnetorques

        # Auxiliary variables
        exp_PTt
        exp_Plt
        exp_Ra
        exp_ha
        exp_hr
        exp_Rhr
        exp_mbt
        exp_mpt
        exp_mAt
        exp_mTt
        exp_mPt
        exp_mSt
        exp_mct
        exp_d

        # Disjunctive
        exp_mMPt >= 0.0

        # Component related variables
        x_P2, Bin
        x_M, Bin
        x_T[1:n_T], Bin
        x_b[1:n_b], Bin
        x_p[1:n_p], Bin
        x_A[1:n_A], Bin
    end

    # Catalog selections
    @constraints m begin
        x_P2 + x_M == 1

        sum(x_T) == 1 # transmitter catalog
        G_T == dot(x_T, G_Ti)
        m_T == dot(x_T, m_Ti)
        P_T == dot(x_T, P_Ti)
        #G_T == dot(x_T, G_Ti)
        #m_T == ρ_T + 3/2*D_T

        sum(x_b) == 1 # battery catalog
        E_b == dot(x_b, E_bi)
        m_b == dot(x_b, m_bi)
        #m_b == ρ_b + E_b

        sum(x_p) == 1 # payload catalog
        D_p == dot(x_p, D_pi)
        m_p == dot(x_p, m_pi)
        #m_p == ρ_p + 3/2*D_p

        sum(x_A) == 1 # solar panel catalog
        ρ_A == dot(x_A, ρ_Ai)
        η_A == dot(x_A, η_Ai)
    end
    # Piecewiselinear function values from data table
    ρ = piecewiselinear(m, h_ρ, h_ρi, ρi)
    H = piecewiselinear(m, h_ρ, h_ρi, Hi)

    # Nonconvex extended formulation piecewiselinear approx
    brk_log = (v_min, v_max, num_pts) -> log.(linspace(exp(v_min), exp(v_max), num_pts))
    pwgraph_exp_d = piecewiselinear(m, d, brk_log(d_min, d_max, n_pts), exp) # convex
    pwgraph_exp_e = piecewiselinear(m, e, brk_log(e_min, e_max, n_pts), exp) # convex
    pwgraph_exp_g = piecewiselinear(m, g, brk_log(g_min, g_max, n_pts), exp) # convex
    fhR = h_val -> log(acos(1/(exp(h_val - R) + 1))) # instead of linearized
    pwgraph_fhR = piecewiselinear(m, h, brk_log(h_min, h_max, n_pts), fhR) # concave
    ahr = h_val -> log(exp(h_val) + exp(R))
    pwgraph_a = piecewiselinear(m, h, brk_log(h_min, h_max, n_pts), ahr) #convex
    # Lifetime
    pwgraph_Ln = piecewiselinear(m, exp_Ln, linspace(exp_Ln_min, exp_Ln_max, n_pts), log) # concave
    pwgraph_Lp = piecewiselinear(m, exp_Lp, linspace(exp_Lp_min, exp_Lp_max, n_pts), log) # concave

    # Convex extended formulation constraints
    @NLconstraints m begin
        exp(2*h - 2*r) <= exp_hr
        exp(log(2) + R + h - 2*r) <= exp_Rhr

        exp(d) <= exp_d
        log(π) + g <= log(acos(1/(exp(h - R) + 1)))

        exp(m_P2 - m_t) <= exp_mMPt
        exp(m_M - m_t) <= exp_mMPt
        (x_P2 + 1e-3)*exp((m_P2 - m_t)/(x_P2 + 1e-3)) <= exp_mMPt
        (x_M + 1e-3)*exp((m_M - m_t)/(x_M + 1e-3)) <= exp_mMPt

        exp(P_T - P_t) <= exp_PTt
        exp(P_l - P_t) <= exp_Plt

        exp(m_b - m_t) <= exp_mbt
        exp(m_p - m_t) <= exp_mpt
        exp(m_A - m_t) <= exp_mAt
        exp(m_T - m_t) <= exp_mTt
        exp(m_P - m_t) <= exp_mPt
        exp(m_S - m_t) <= exp_mSt
        exp(m_c - m_t) <= exp_mct
    end

    # Linear constraints
    @constraints m begin
        # Power and communications
        P_t - d == A + η_A + Q
        E_b >= P_t - d + T
        P_T <= P_t
        EN + L + k + T_s + R + log(2π) + N + B + log(4) + 2*r <= P_T + G_r + X_r + g + T + G_T + 2*(λ_c-log(π))
        # Payload performance
        X_R >= X_r
        X_r == log(1.22) + h + λ_v - D_p
        # Orbit
        # g == α_1 + γ_1*(h - R) # linearized
        a == pwgraph_a
        T - log(2π) == (3*a - μ)/2

        # Mass budgets
        m_A == ρ_A + A
        #m_P == ρ_P - h
        m_S == η_S + m_t

        # lifetime
        pwgraph_Ln + log(3600*24*365) == T + H + m_t - log(2π) - C_D - A - 2*a - ρ
        pwgraph_Lp + log(3600*24*365) == m_P + I_sp + G + a - log(0.5) - C_D - A - ρ - μ
        exp_Ln + exp_Lp >= exp_Lt_min
        h_ρ == h

        # From convex constraints
        exp_PTt + exp_Plt <= 1
        #exp_Ra + exp_ha <= 1
        exp_hr + exp_Rhr <= 1
        exp_mbt + exp_mpt + exp_mPt + exp_mAt + exp_mTt + exp_mSt + exp_mct + exp_mMPt<= 1

        # From nonconvex constraints
        exp_d <= pwgraph_exp_d
        pwgraph_exp_e + exp_d == 1
        exp_d <= pwgraph_exp_g + 1/2
        log(π) + g >= pwgraph_fhR

        # Momentum budgets
        T_g == log(3) + μ + c_W - log(2) - 3*a
        m_P2 == ρ_P2 + T + T_g + log(1/4*sqrt(2)/2*3*365) + log(exp_Lt_min) - I_sp - G
        m_M == ρ_M + T_g
    end

    # Minimize total mass
    @objective(m, Min, m_t)

    # Solve
    status = solve(m)
    D = Dict(
        "status"=> status,
        "m_t" => exp(getvalue(m_t)),
        "x_p" => find(i -> (i > 0.5), getvalue(x_p)),
        "x_R" => exp(X_R),
        "x_r" => exp(getvalue(X_r)),
        "x_B" => find(i -> (i > 0.5), getvalue(x_b)),
        "m_B" => exp(getvalue(m_b)),
        "m_S" => exp(getvalue(m_S)),
        "x_T" => find(i -> (i > 0.5), getvalue(x_T)),
        "m_T" => exp(getvalue(m_T)),
        "G_T" => exp(getvalue(G_T)),
        "data" => 2*π*exp(R)/exp(getvalue(X_r))*exp(B)*exp(N),
        "rate" => 2*π*exp(R)/exp(getvalue(X_r))*exp(B)*exp(N)/exp(getvalue(T)),
        "x_A" => find(i -> (i > 0.5), getvalue(x_A)),
        "A" => exp(getvalue(A)),
        "m_A" => exp(getvalue(m_A)),
        "Ln" => getvalue(exp_Ln),
        "Lp" => getvalue(exp_Lp),
        "Lt" => getvalue(exp_Ln)+getvalue(exp_Lp),
        "h" => exp(getvalue(h))/1000,
        "a" => exp(getvalue(a))/1000-exp(R),
        "d" => exp(getvalue(d)),
        "e" => exp(getvalue(e)),
        "g" => exp(getvalue(g)),
        "T" => exp(getvalue(T))/60,
        "ρ" => exp(getvalue(ρ)),
        "m_P" => exp(getvalue(m_P)),
        "x_disj" => getvalue(x_P2) > 0.5 ? "P" : "M",
        "T_g" => exp(getvalue(T_g)),
        "m_P2" => exp(getvalue(m_P2)),
        "m_M" => exp(getvalue(m_M))
    )

    # Store data
    stringdata = json(D,4)
    print(stringdata)

    bit_value = round(exp(B))
    filename = "resolution_$(X_R_original)_lifespan_$(exp_Lt_min)_bits_$(bit_value).json"
    # open(filename, "w") do f
    #         write(f, stringdata)
    #      end
end


#using MINLPOA
#global_solver = MINLPOASolver(log_level=1, mip_solver=mip_solver)

# Run
#global_model(global_solver, 100)

# # Define solvers
using Ipopt
using Pajarito
#
global_solver = PajaritoSolver(
    mip_solver=mip_solver,
    cont_solver=IpoptSolver(print_level=0),
    mip_solver_drives=true,
    log_level=1,
    rel_gap=1e-7)
#
# # Run
cases = [[50, 5, 8], [10, 3, 8], [7.5, 5, 8], [5, 3, 8], [5, 3, 16]]
for (X_R, lifetime, B) in cases
    global_model(global_solver, 100, X_R, lifetime, B)
end

## =======================
# Chris old code
## =======================

# MILP solver
# using CPLEX
# mip_solver = CplexSolver(CPX_PARAM_SCRIND=1, CPX_PARAM_EPINT=1e-9, CPX_PARAM_EPRHS=1e-9, CPX_PARAM_EPGAP=1e-7)
#using Gurobi
#mip_solver = Gurobi.GurobiSolver(OutputFlag=1, IntFeasTol=1e-9, FeasibilityTol=1e-9, MIPGap=1e-7)

# MINLP solver
#using DaChoppa
#global_solver = DaChoppaSolver(log_level=1, mip_solver=mip_solver)

# using Ipopt
# using Pavito
# global_solver = PavitoSolver(
#     mip_solver=mip_solver,
#     cont_solver=IpoptSolver(print_level=0),
#     mip_solver_drives=true,
#     log_level=1,
#     rel_gap=1e-7,
#     )

# Run
#global_model(global_solver, 100)

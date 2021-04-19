#=
Satellite optimization problem from from [Norheim, 2020]
Known optimum is 3.15 kg mass. 
=#


function satellite(solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    m = Model(solver=solver)

    # Variables 
    @variables m begin
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

    # Available components
    m[:D_P] = [0.025, 0.075, 0.1]
    m[:m_P] = [0.150, 1.75, 3.0]

    @variable(m, x[1:8])
    @objective(m, Min, x[8])
    gm = GlobalModel(model = m, name = "satellite")

    return gm
end
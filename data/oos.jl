function oos()
    rE = 6378137 # radius of Earth (m)
    mu = 3.986e14 # gravitational constant

    n_sats = 11 # number of satellites
    n_sers = 1 # number of services
    true_anomalies = collect(1:nsats).*2*pi/n_sats
    phasing_ratios = (2pi .- true_anomalies) ./ (2pi)

    alt_sat = 780e3 # altitude of satellites (m)
    r_sat = rE + alt_sat
    period_sat = 2pi*sqrt(r_sat^3/mu)

    maximum_time = 10*3600*24*365 # 10 years in seconds

    m = Model(Ipopt.Optimizer)
    @variable(m, r_orbit[1:n_sats] >= rE)
    @variable(m, period_orbit[1:n_sats] >= 0)
    @variable(m, dt_orbit[1:n_sats] >= 0)
    @variable(m, N_orbit[1:n_sats] >= 0)

    @NLconstraint(m, [i=1:n_sats], period_orbit[i] == 2pi*sqrt(r_orbit[i]^3 / mu))
    @NLconstraint(m, )
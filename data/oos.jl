using Parameters

@with_kw mutable struct oos_params
    g = 9.81
    mu = 3.986e14 # gravitational constant
    rE = 6378137 # radius of Earth (m)
    dry_mass = 500 # empty mass of satellite((kg))
    Isp = 230     # Specific impulse
    alt_sat = 780e3 # altitude of satellites (m)
    n_sats = 11     # number of satellites (must be odd for this case)
    true_anomalies = circshift(collect(1:n_sats).*2*pi/n_sats .- 2pi*6/n_sats, Int((n_sats-1)/2)+1)
    fuel_required = circshift(collect(100 .+ range(-20, stop=40, length=n_sats)), 4) # dummy example 
    r_sat = rE + alt_sat
    period_sat = 2pi*sqrt(r_sat^3/mu)
    r_orbits = rE .+ 1e3 .* collect(680:10:890)
    orbital_periods = [2pi*sqrt(r^3/mu) for r in r_orbits]
    transfer_time = [2*pi*sqrt((r_sat + r)^3/(8*mu)) for r in r_orbits]
    maximum_time = 0.75*3600*24*365 # 0.75 years in seconds
end

function oos_gm!(op = oos_params())
    n = op.n_sats

    # Making sure the orbits don't include the actual orbit of the satellite
    op.r_orbits = filter!(r -> r != op.r_sat, op.r_orbits)  
    op.orbital_periods = [2pi*sqrt(r^3/op.mu) for r in op.r_orbits]
    op.transfer_time = [2*pi*sqrt((op.r_sat + r)^3/(8*op.mu)) for r in op.r_orbits]

    # Precomputing fractional changes in mass for different orbits
    # NOTE: don't even include the orbital mass changes. 
    # dmass_entry = []
    # dmass_exit = []
    # for r in op.r_orbits
    #     if r >= op.r_sat
    #         push!(dmass_entry, 1/exp(1/(op.g*op.Isp) * 
    #                             sqrt(op.mu/r)*(sqrt(2op.r_sat/(op.r_sat + r)) -1)))
    #         push!(dmass_exit, 1/exp(1/(op.g*op.Isp) * 
    #                             sqrt(op.mu/op.r_sat)*(1 - sqrt(2r/(op.r_sat + r)))))
    #     else
    #         push!(dmass_entry, exp(1/(op.g*op.Isp) * 
    #                             sqrt(op.mu/r)*(sqrt(2op.r_sat/(op.r_sat + r)) -1))) #TODO Check these, don't make sense.
    #         push!(dmass_exit, exp(1/(op.g*op.Isp) * 
    #                             sqrt(op.mu/op.r_sat)*(1 - sqrt(2r/(op.r_sat + r)))))
    #     end
    # end

    # Starting a JuMP.Model
    m = Model(CPLEX_SILENT)

    @variable(m, dmass_entry[i=1:n-1] >= 1)
    @variable(m, dmass_exit[i=1:n-1] >= 1)
    # Locations xx, and sat_order (mostly for post_processing)
    @variable(m, n >= sat_order[i=1:n] >= 1)
    @variable(m, xx[1:n, 1:n], Bin)        # transfer occurs from row index to column index

    # Maneuver time variables
    @variable(m, 1000 >= N_orbit[1:n-1] >= 0)
    @variable(m,  maximum(abs.(op.period_sat .- op.orbital_periods)) >= dt_orbit[1:n-1] >= -maximum(abs.(op.period_sat .- op.orbital_periods)))
    @variable(m, maximum(op.transfer_time) >= dt_transfer_orbit[1:n-1] >= minimum(op.transfer_time))
    @variable(m, maximum(op.r_orbits) >= r_orbit[1:n-1] >= minimum(op.r_orbits))
    @variable(m, maximum(op.orbital_periods) >= period_orbit[1:n-1] >= minimum(op.orbital_periods))

    # Program can pick the best starting satellite as well. 
    @constraint(m, [i=1:n], sat_order[i] == sum(xx[i, :] .* collect(1:n)))
    @constraint(m, [i=1:n], sum(xx[i, :]) == 1) # Exactly one transfer every time (same as each satellite visited once)
    @constraint(m, [i=1:n], sum(xx[:, i]) == 1)   # At most one transfer to each satellite

    # # True anomaly computation from satellite order
    @variable(m, pi + 1e-5 >= ta[1:n-1] >= -pi -1e-5)
    @variable(m, -1 <= rotdummy[1:n-1] <= 1, Int) # for rotations
    @constraint(m, [i=1:n-1], ta[i] == sum(op.true_anomalies .* xx[i+1,:]) - 
                                        sum(op.true_anomalies .* xx[i,:]))

    # Fuel consumption (Watch these arbitrary upper bounds)
    @variable(m, 2500 >= masses[1:n-1, 1:5] >= op.dry_mass)
    @variable(m, 3000 >= wet_mass >= op.dry_mass)

    # Orbit choice, and time and fuel cost
    @variable(m, 1.15 >= fractional_dmasses[1:n-1, j=1:5] >= 1) # reduction in masses in maneuvers
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 1] == dmass_entry[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 2] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 3] == dmass_exit[i])
    @constraint(m, [i=1:n-1], fractional_dmasses[i, 4] == dmass_entry[i])

    # Computing required fuel depending on transfer schedule
    @variable(m, maximum(op.fuel_required) >= fuel_needed[1:n] >= minimum(op.fuel_required))
    @constraint(m, [i=1:n], fuel_needed[i] == sum(xx[i, :] .* op.fuel_required))

    # Mass conservation constraints (linear)
    @constraint(m, wet_mass >= masses[1, 1] + fuel_needed[1])
    @constraint(m, [i=1:n-2], masses[i, 5] >= masses[i+1, 1] + fuel_needed[i+1])
    @constraint(m, masses[n-1, 5] >= op.dry_mass + fuel_needed[n])
    for i=1:n-1
        @constraint(m, masses[i, 1] >= masses[i, 2])
        @constraint(m, masses[i, 2] >= masses[i, 3])
        @constraint(m, masses[i, 3] >= masses[i, 4])
        @constraint(m, masses[i, 4] >= masses[i, 5])
    end

    # Maneuver time constraints (linear)
    @constraint(m, [i=1:n-1], dt_orbit[i] == period_orbit[i] - op.period_sat)
    @variable(m, t_maneuver[1:n-1] >= 0)
    @constraint(m, sum(t_maneuver) <= op.maximum_time)

    @objective(m, Min, wet_mass)

    gm = GlobalModel(model = m, name = "oos")
    # Mass conservation constraints (bilinear)
    add_nonlinear_constraint(gm, :((masses, fractional_dmasses) -> masses[1, 2] * fractional_dmasses[1,1]), 
        vars = [masses[1, 2], fractional_dmasses[1,1]], dependent_var = masses[1, 1],
        name = "mass_fraction")
    for j=2:4
        add_linked_constraint(gm, gm.bbls[end], [masses[1, j+1], fractional_dmasses[1,j]], masses[1,j])
    end
    for i=2:n-1
        for j=1:4
            add_linked_constraint(gm, gm.bbls[end], [masses[i, j+1], fractional_dmasses[i,j]], masses[i,j])
        end
    end

    # Maneuver time constraints (nonlinear + bilinear)
    add_nonlinear_constraint(gm, :(r_orbit -> 2*pi*sqrt(r_orbit[1]^3/$(op.mu))), 
        vars = [r_orbit[1]], dependent_var = period_orbit[1],
        name = "orbital_period")
    [add_linked_constraint(gm, gm.bbls[end], [r_orbit[i]], period_orbit[i]) for i=2:n-1]

    add_nonlinear_constraint(gm, :(r_orbit -> 2*pi*sqrt(($(op.r_sat) + r_orbit[1])^3/(8*$(op.mu)))), 
        vars = [r_orbit[1]], dependent_var = dt_transfer_orbit[1],
        name = "transfer_time")
    [add_linked_constraint(gm, gm.bbls[end], [r_orbit[i]], dt_transfer_orbit[i]) for i=2:n-1]

    add_nonlinear_constraint(gm, :((ta, dt_orbit, N_orbit) -> maximum([N_orbit[1]*dt_orbit[1] - ta[1] * $(op.period_sat), 
        N_orbit[1]*dt_orbit[1] - ta[1] * $(op.period_sat)])), 
        vars = [ta[1], dt_orbit[1], N_orbit[1]],
        name = "orbital_revolutions")
    [add_linked_constraint(gm, gm.bbls[end], [ta[i], dt_orbit[i], N_orbit[i]]) for i=2:n-1]

    add_nonlinear_constraint(gm, :((dt_transfer_orbit, N_orbit, period_orbit) -> 
            dt_transfer_orbit[1] + N_orbit[1] * period_orbit[1]), 
            vars = [dt_transfer_orbit[1], N_orbit[1], period_orbit[1]], dependent_var = t_maneuver[1],
            name = "maneuver_time")
    [add_linked_constraint(gm, gm.bbls[end], [dt_transfer_orbit[i], N_orbit[i], period_orbit[i]], t_maneuver[i])
        for i=2:n-1]

    set_param(gm, :ignore_accuracy, true)

    return gm
end
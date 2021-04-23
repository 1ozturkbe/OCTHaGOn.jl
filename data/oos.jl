using JuMP, Gurobi

# function oos()
g = 9.81
rE = 6378137 # radius of Earth (m)
m_empty = 500 # empty mass of satellite((kg))
mu = 3.986e14 # gravitational constant
Isp = 230     # Specific impulse

n_sats = 11 # number of satellites (must be odd for this case)
n = n_sats
halfpoint = Int((n_sats-1)/2)
n_sers = 1 # number of services
true_anomalies = circshift(collect(1:n_sats).*2*pi/n_sats .- 2pi*6/n_sats, Int((n_sats-1)/2)+1)

fuel_required = circshift(collect(100 .+ range(-20, stop=40, length=n_sats)), 4) # dummy example 

alt_sat = 780e3 # altitude of satellites (m)
r_sat = rE + alt_sat
period_sat = 2pi*sqrt(r_sat^3/mu)

potential_orbits = rE .+ 1e3 .* collect(680:10:880)
potential_orbits = filter!(r -> r != r_sat, potential_orbits)  
n_orbits = length(potential_orbits)
orbital_periods = [2pi*sqrt(r^3/mu) for r in potential_orbits]
dmass_entry = []
dmass_exit = []
for r in potential_orbits
    if r >= r_sat
        push!(dmass_entry, 1/exp(1/(g*Isp) * sqrt(mu/r)*(sqrt(2r_sat/(r_sat + r)) -1)))
        push!(dmass_exit, 1/exp(1/(g*Isp) * sqrt(mu/r_sat)*(1 - sqrt(2r/(r_sat + r)))))
    else
        push!(dmass_entry, exp(1/(g*Isp) * sqrt(mu/r)*(sqrt(2r_sat/(r_sat + r)) -1))) #TODO Check these, don't make sense.
        push!(dmass_exit, exp(1/(g*Isp) * sqrt(mu/r_sat)*(1 - sqrt(2r/(r_sat + r)))))
    end
end

transfer_time = [2*pi*sqrt((r_sat + r)^3/(8*mu)) for r in potential_orbits]
maximum_time = 10*3600*24*365 # 10 years in seconds

m = Model(Gurobi.Optimizer)
set_optimizer_attribute(m, "NonConvex", 2)

# Transfers xx
# Always start by fueling the satellite that needs most fuel
max_idx = findall(x -> x == maximum(fuel_required), fuel_required) 
@variable(m, n_sats >= sat_order[i=1:n] >= 1)
@variable(m, xx[1:n-1, 1:n_sats], Bin)        # transfer occurs from row index to column index

# Program can pick the best starting satellite as well. 
@constraint(m, sat_order[1] == max_idx[1])
@constraint(m, [i=2:n], sat_order[i] == sum(xx[i-1, :] .* collect(1:n)))
@constraint(m, [i=1:n-1], sum(xx[i, :]) == 1) # Exactly one transfer every time (same as each satellite visited once)
@constraint(m, [i=1:n], sum(xx[:, i]) <= 1)   # At most one transfer to each satellite
@constraint(m, sum(xx[:, max_idx[1]]) == 0)

# # True anomaly computation from satellite order
@variable(m, ta[1:n-1])
@variable(m, rotdummy[1:n-1], Int) # for rotations
@constraint(m, [i=1:n-1], ta[i] == (sat_order[i+1] - sat_order[i] + 
                                    rotdummy[i] * n_sats)*2pi/n_sats)

# Fuel consumption
@variable(m, masses[1:n_sats-1, 1:5] >= m_empty)
@variable(m, wet_mass >= m_empty)

# Nonlinear versions
# @NLconstraint(m, fuel_consump[1] >= g*Isp*log(wet_mass / end_mass[1]))
# @NLconstraint(m, [i=2:n], fuel_consump[i] >= g*Isp*log(end_mass[i] / end_mass[i-1]))

# Orbit choice, and time and fuel cost
@variable(m, y[i=1:n_sats-1, j=1:n_orbits], Bin) # i is the maneuver order, j is the orbit choice 
@constraint(m, [i=1:n-1], sum(y[i, :]) == 1)
@variable(m, fractional_dmasses[1:n_sats-1, j=1:5] >= 0) # reduction in masses in maneuvers

@constraint(m, [i=1:n-1], fractional_dmasses[i, 1] == (sum(y[i,:] .* dmass_entry)))  
@constraint(m, [i=1:n-1], fractional_dmasses[i, 2] == (sum(y[i,:] .* dmass_exit))) 
@constraint(m, [i=1:n-1], fractional_dmasses[i, 3] == 0)
@constraint(m, [i=1:n-1], fractional_dmasses[i, 4] == (sum(y[i,:] .* dmass_exit))) 
@constraint(m, [i=1:n-1], fractional_dmasses[i, 5] == (sum(y[i,:] .* dmass_entry))) 

# Computing required fuel depending on transfer schedule
@variable(m, fuel_needed[1:n])
@constraint(m, fuel_needed[1] == maximum(fuel_required))
@constraint(m, [i=2:n], fuel_needed[i] == sum(xx[i-1, :] .* fuel_required))

# Mass conservation constraints (bilinear)
@constraint(m, wet_mass >= maximum(fuel_required) + masses[1, 1] * fractional_dmasses[1,1])
for i=1:n_sats-1
    @constraint(m, masses[i, 1] >= masses[i, 2] * fractional_dmasses[i, 2])
    @constraint(m, masses[i, 2] >= masses[i, 3] + fuel_needed[i])
    @constraint(m, masses[i, 3] >= masses[i, 4] * fractional_dmasses[i, 4])
    @constraint(m, masses[i, 4] >= masses[i, 5] * fractional_dmasses[i, 5])
end
@constraint(m, [i=1:n_sats-2], masses[i, 5] >= masses[i+1, 1] * fractional_dmasses[i+1,1])

# Maneuver time constraints
@variable(m, N_orbit[1:n_sats-1] >= 0)
@variable(m, dt_orbit[1:n_sats-1])
@variable(m, period_orbit[1:n_sats-1] >= 0)
@constraint(m, [i=1:n-1], period_orbit[i] == sum(y[i, :] .* orbital_periods))
@constraint(m, [i=1:n-1], dt_orbit[i] == period_orbit[i] - period_sat)
@constraint(m, [i=1:n-1], N_orbit[i]*dt_orbit[i] >= ta[i] * period_sat)
@constraint(m, [i=1:n-1], N_orbit[i]*dt_orbit[i] >= -ta[i] * period_sat)

@variable(m, t_maneuver[1:n_sats-1] >= 0) # Maneuvering time (s)
@constraint(m, [i=1:n-1], t_maneuver[i] == sum(y[i,:] .* transfer_time) + N_orbit[i] * period_orbit[i])

@constraint(m, sum(t_maneuver) <= maximum_time)

@objective(m, Min, wet_mass)

optimize!(m)

println("Orbit altitudes (km) : $(round.([(sum(getvalue.(y)[i,:] .* potential_orbits) - rE)/1e3 for i=1:n_sats-1], sigdigits=5))")
println("Satellite order: $(Int.(round.(getvalue.(sat_order))))")
println("True anomalies (radians): $(round.(getvalue.(ta), sigdigits=3))")
println("Orbital revolutions: $(round.(abs.(period_sat .* getvalue.(ta) ./ getvalue.(dt_orbit)), sigdigits=5)))")
println("Time for maneuvers (days): $(round.(getvalue.(t_maneuver)./(24*3600), sigdigits=3))")
println("Total mission time (years): $(sum(getvalue.(t_maneuver))/(24*3600*365))")


# Transfer debugging
# m = Model(Gurobi.Optimizer)
# max_idx = findall(x -> x == maximum(fuel_needed), fuel_needed)
# @variable(m, xx[1:n-1, 1:n_sats], Bin)        # transfer occurs from row index to column index
# @constraint(m, [i=1:n-1], xx[i, i] == 0)      # Cannot self transfer
# @constraint(m, [i=1:n-1], sum(xx[i, :]) == 1) # Exactly one transfer every time (same as each satellite visited once)
# @variable(m, pi >= ta[1:n-1] >= -pi)            # true anomalies
# max_idx = findall(x -> x == maximum(fuel_needed), fuel_needed)
# @constraint(m, [i=1:n-1], ta[i] == sum(circshift(true_anomalies, i-1) .* xx[i, :]))
# # @variable(m, dt[1:n_sats] >= 0) # Actual period time differences (s)


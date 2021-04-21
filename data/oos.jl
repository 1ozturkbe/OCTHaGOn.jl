using JuMP, Gurobi

# function oos()
g = 9.81
rE = 6378137 # radius of Earth (m)
m_empty = 500 # empty mass of satellite((kg))
mu = 3.986e14 # gravitational constant
Isp = 230     # Specific impulse

n_sats = 11 # number of satellites
n = n_sats
n_sers = 1 # number of services
true_anomalies = collect(1:n_sats).*2*pi/n_sats
fuel_needed = 100 .* ones(n)
phasing_ratios = (2pi .- true_anomalies) ./ (2pi)

alt_sat = 780e3 # altitude of satellites (m)
r_sat = rE + alt_sat
period_sat = 2pi*sqrt(r_sat^3/mu)

potential_orbits = rE .+ 1e3 .* collect(600:20:760)  
n_orbits = length(potential_orbits)
orbital_periods = [2pi*sqrt(r^3/mu) for r in potential_orbits]
dmass_entry = [exp(1/(g*Isp) * sqrt(mu/r)*(sqrt(2r_sat/(r_sat + r)) -1)) for r in potential_orbits]
dmass_exit = [exp(1/(g*Isp) * sqrt(mu/r_sat)*(1 - sqrt(2r/(r_sat + r)))) for r in potential_orbits]
transfer_time = [2*pi*sqrt((r_sat + r)^3/(8*mu)) for r in potential_orbits]
maximum_time = 10*3600*24*365 # 10 years in seconds

m = Model(Gurobi.Optimizer)
set_optimizer_attribute(m, "NonConvex", 2)

# Fuel consumption
@variable(m, masses[1:n_sats, 1:4] >= m_empty)
@variable(m, wet_mass >= m_empty)
# Nonlinear versions
# @NLconstraint(m, fuel_consump[1] >= g*Isp*log(wet_mass / end_mass[1]))
# @NLconstraint(m, [i=2:n], fuel_consump[i] >= g*Isp*log(end_mass[i] / end_mass[i-1]))

# Orbit choice, and time and fuel cost
@variable(m, y[i=1:n_sats, j=1:n_orbits], Bin) # i is the satellite, j is the orbit choice 
@constraint(m, [i=1:n], sum(y[i, :]) == 1)
@variable(m, fractional_dmasses[1:n_sats, j=1:4] >= 0) # reduction in masses in maneuvers

@constraint(m, [i=1:n], fractional_dmasses[i, 1] == (sum(y[i,:] .* dmass_entry)))  
@constraint(m, [i=1:n], fractional_dmasses[i, 2] == (sum(y[i,:] .* dmass_exit))) 
@constraint(m, [i=1:n], fractional_dmasses[i, 3] == (sum(y[i,:] .* dmass_entry))) 
@constraint(m, [i=1:n], fractional_dmasses[i, 4] == (sum(y[i,:] .* dmass_exit))) 


@constraint(m, wet_mass >= masses[1, 1] * fractional_dmasses[1,1])
for i=1:n_sats-1
    @constraint(m, masses[i, 1] >= masses[i, 2] * fractional_dmasses[i,2])
    @constraint(m, masses[i, 2] >= masses[i, 3] * fractional_dmasses[i,3] + 200) #TODO fix
    @constraint(m, masses[i, 3] >= masses[i, 4] * fractional_dmasses[i,4])
    @constraint(m, masses[i, 4] >= masses[i+1, 1] * fractional_dmasses[i+1,1])
end
@constraint(m, masses[n, 1] >=  masses[n, 2] * fractional_dmasses[n,2])
@constraint(m, masses[n, 2] >=  masses[n, 3] * fractional_dmasses[n,3] + 200) #TODO fix
@constraint(m, masses[n, 3] >= masses[n, 4] * fractional_dmasses[n,4])

# @constraint(m, [i=1:n], fuel_consump[i] >= 1/(g*Isp) * (sum(y[i,:] .* orbital_deltav_entry) + 
#                                                         sum(y[i,:] .* orbital_deltav_exit)))

@variable(m, t_maneuver[1:n_sats] >= 0) # Maneuvering time (s)
@constraint(m, [i=1:n], t_maneuver[i] == sum(y[i,:] .* transfer_time))

@objective(m, Min, sum(t_maneuver)/100 + wet_mass)


# @variable(m, period_orbit[1:n_sats] >= 0)
# @variable(m, dt_orbit[1:n_sats] >= 0)
# @variable(m, N_orbit[1:n_sats] >= 0)

# @variable(m, x[1:n_sats, 1:n_sats], Bin) # order of servicing, ith sat in jth order
# @variable(m, ta[1:n_sats]) # Actual true anomalies
# @variable(m, dt[1:n_sats] >= 0) # Actual period time differences (s)

# @variable(m, t_maneuver[1:n_sats] >= 0) # Maneuvering time (s)
# @NLconstraint(m, [i=1:n_sats], period_orbit[i] == 2pi*sqrt(r_orbit[i]^3 / mu))
# @constraint(m, [i=1:n], N_orbit[i] == dt_orbit)
# @NLconstraint(m, [i=1:n_sats], t_maneuver[i] >= 
#                 2pi*sqrt((r_orbit + r_sat)^3/ (8mu)) + N_orbit[i] * dt_orbit[i])
# @constraint(m, sum(t_maneuver) <= maximum_time)

# @variable(m, m[1:n_sats] >= m_empty)

# @constraint(m, [i = 1:n], sum(x[:, i]) == 1)
# @constraint(m, [i = 1:n], sum(x[i, :]) == 1)
# @constraint(m, [j=1:n], ta[j] == sum(true_anomalies .* x[:, j]))
# @constraint(m, [j=1:n], dt_orbit[j] == period_sat * (1 - ta[j]/(2pi)))

# @NLconstraint(m, )
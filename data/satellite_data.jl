#using Distributions

μ = 3.986005e14 # m^3/s^2 standard gravitational parameter
R = 6378e3 # m
Q = 1367 # W/m^2
EN = 14
D_r = 5.3 # m
L = 9.772372209558107 # example from SMAD in DB: 8.5+1+0.1+0.3
k = 1.38064852e-23 # J/K Boltzman constant R/Na
T_s = 135 # K
B = 8 # bit
N = 2e3 # pixel width
η = 0.55
λ_v = 500e-9 # m
f = 2.2e9 # Hz
c = 2.998e8 # m/s
#η_A = 0.29
#ρ_A = 10 # kg/m^2
#ρ_p = 100 # kg/m^1.5
#ρ_T = 2 # kg/m^1.5
ρ_P = 500e3 # kg*m
ρ_b = 0.002e-3 # kg/J
P_l = 5 # W

#X_r = 7.5 # m

m_c = 0.2 # kg required weight
η_S = 0.2 # 20% of total mass is structural mass
C_D = 2.2 # coefficient of drag
G = 9.81 # m/s^2 gravitational constant
I_sp = 70 # s impulse specific constant
w_W = 1000 # radians/s for maximum angular velocity of reaction wheel
c_W = 1 # constant to make gravity gradient significant enough to count
ρ_M = 11e4  # MTQ
ρ_P2 = 9/0.1 # acs
M_B = 7.96e15


λ_c = c - f
G_r = η + 2*(log(π) + D_r - λ_c)
#print(exp(G_r))
#G_T/2 - η/2 - log(π) + λ_c = D_r

T2 = 24*3600

m_max = 7
m_min = 0.01

exp_Ln_min = 0.1
exp_Ln_max = 200

exp_Lp_min = 0.0001
exp_Lp_max = 25.0

h_min = 1e5 # orbit altitude
h_max = 2e6
a_min = log(exp(h_min)+exp(R))
a_max = log(exp(h_max)+exp(R))
k_max = T2/(2π*sqrt(exp(a_min)^3/exp(μ)))
k_min = T2/(2π*sqrt(exp(a_max)^3/exp(μ)))

d_min = 0.1 # daylight fraction of orbit
d_max = 0.9

e_min = 0.1 # eclipse fraction of orbit
e_max = 0.9

exp_RhR_min = 1/(exp(h_min - R) + 1)
exp_RhR_max = 1/(exp(h_max - R) + 1)

g_min = acos(exp_RhR_min)/pi # ground station viewing fraction of orbit
g_max = acos(exp_RhR_max)/pi

dBtoLinear(db) = 10^(db/10)

#srand(124)
n_T = 2 # transmitter catalog
#n_T = 100
#D_Ti = log.([0.07, 0.14])
#G_r = η + 2*(log(π) + D_Ti - λ_c)
m_Ti = [0.053, 0.300]
#m_Ti = log.(rand(Uniform(0.01,1),n_T))
P_Ti = [10, 10]
#P_Ti = log.(rand(Uniform(5,15),n_T))
G_Ti = [dBtoLinear(11), dBtoLinear(16.5)]
#G_Ti = log.([dBtoLinear(x) for x in rand(Uniform(10,20),n_T)])

n_b = 5 # battery catalog
#n_b = 100
E_bi = [138600, 144000, 144000, 165600, 1607040]
#E_bi = log.(rand(Uniform(1e4, 1e6),n_b))
m_bi = [0.270, 0.310, 0.355, 0.710, 3.95]
#m_bi = log.(rand(Uniform(0.1, 5),n_b))

n_p = 3 # payload catalog
#n_p = 100
D_pi = [0.025, 0.075, 0.1] #0.025
#D_pi = log.(rand(Uniform(0.01, 0.075),n_p))
m_pi = [0.080, 1.75, 3.0] #0.080
P_pi = [0.5, 1, 2]
#m_pi = log.(rand(Uniform(0.1, 5),n_p))

n_A = 2 # solar panel catalog
#n_A = 100
#5.35 = 0.043/(0.098*0.082)
ρ_Ai = [5.35, 0.84]
#ρ_Ai = log.(rand(Uniform(0.85, 6),n_A))
η_Ai = [0.305, 0.275]
#η_Ai = log.(rand(Uniform(0.15, 0.31),n_A))


# piecewise linear table
h_ρi = log(1e3) .+ log.([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500])
ρi = log.([4.79e-07, 1.81e-09, 2.53e-10, 6.24e-11, 1.95e-11, 6.98e-12, 2.72e-12, 1.13e-12, 4.89e-13, 2.21e-13, 1.04e-13, 5.15e-14, 2.72e-14, 1.55e-14, 9.63e-15, 6.47e-15, 4.66e-15, 3.54e-15, 2.79e-15, 1.11e-15, 5.21e-16])
Hi = log(1e3) .+ log.([5.9, 25.5, 37.5, 44.8, 50.3, 54.8, 58.2, 61.3, 64.5, 68.7, 74.8, 84.4, 99.3, 121, 151, 188, 226, 263, 296, 408, 516])
;

# operations simulation
orbits = 5
Nperorbit = 20
Nt = Nperorbit*orbits

# Ground stations
n_gs = 6    # points on the map
n_GS = 5
max_n_gs = 1 # amount of stations we can pick
Random.seed!(100)
lat_g0, lon_g0 =  (0.5-rand(1))*π/2, rand(1)*2π
lat_g = vcat(lat_g0, (0.5-rand(n_gs-1))*π/2)
lon_g = vcat(lon_g0, rand(n_gs-1)*2π)

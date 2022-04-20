""" Hanoi water network model. """
function hanoi_network(oct::Bool = false, friction::String = "HW")
    m = JuMP.Model()
    N = 32
    m[:N] = 32 # Number of nodes
    K = 34
    m[:K] = 34

    sink = [0, 890, 850, 130, 725, 1005, 1350, 550, 525,
            525, 500, 560, 940, 615, 280, 310,
            865, 1345, 60, 1275, 930, 485, 1045, 820, 170,
            900, 370, 290, 360, 360, 105, 805]/3600.0
    source = zeros(m[:N])
    source[1] = sum(sink)
    topology_list = [[0, 1], [1, 2], [2, 3], [2, 19], [2, 18], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [16, 15], [17, 16], [18, 17], [15, 26], [26, 25],
                    [25, 24], [23, 24], [22, 23], [22, 27], [27, 28], [28, 29], [29, 30], [24, 31], [31, 30], [19, 20],
                    [20, 21], [19, 22]]
    topology_dict = Dict(i => topology_list[i] .+ 1 for i = 1:m[:K])
    coordinates = Dict(1 => (5021.20, 1582.17), 2 => (5025.20, 2585.42), 3 => (5874.22, 2588.30), 4 => (6873.11, 2588.30),
                5 => (8103.51, 2585.42), 6 => (8103.51, 3234.67), 7 => (8106.66, 4179.28), 8 => (8106.66, 5133.78),
                9 => (7318.64, 5133.78), 10 => (7319.94, 5831.65), 11 => (7319.94, 6671.19), 12 => (5636.76, 6676.24),
                13 => (6530.63, 5133.78), 14 => (5676.02, 5133.78), 15 => (5021.20, 5133.78), 16 => (5021.20, 4412.36),
                17 => (5021.20, 3868.52), 18 => (5021.20, 3191.49), 19 => (3587.87, 2588.30), 20 => (3587.87, 1300.84),
                21 => (3587.87, 901.29), 22 => (1978.55, 2588.30), 23 => (1975.58, 4084.35), 24 => (1980.46, 5137.63),
                25 => (3077.46, 5137.63), 26 => (3933.52, 5133.78), 27 => (846.04, 2588.20), 28 => (-552.41, 2588.20),
                29 => (-552.38, 4369.06), 30 => (-549.36, 5137.63), 31 => (536.45, 5137.63), 0 => (5360.71, 1354.05))
    L = [100, 1350, 900, 1150, 1450, 450, 850, 850, 800, 950, 1200, 3500, 800, 500, 550, 2730, 1750, 800, 400, 2200,
        1500, 500, 2650, 1230, 1300, 850, 300, 750, 1500, 2000, 1600, 150, 860, 950]

    # Parameters
    rho = 1000 # Density (kg/m^3)
    mu =  8.9e-4 # Viscosity, kg/m/s
    g = 9.81 # Acceleration of gravity, m/s^2
    H_min = 30
    D_min = 0.3048 # Minimum pipe diameter, m
    D_max = 1.016 # Maximum pipe diameter, m
    V_max = 100

    if friction == "HW"
        rough = 130 .* ones(K)
    elseif friction == "DW"        
        rough = 0.26e-6 .* ones(K)
    end

    @variable(m, H[1:N] >= H_min)             # Head
    @constraint(m, H[1] == 100)
    @variable(m, pipeCost[1:K] >= 0)      # Pipe cost
    @variable(m, D_max >= D[1:K] >= D_min)             # Pipe diameter
    @variable(m, 6 >= flow[1:K] >= 0)
    # @variable(m, flow_dir[1:K], Bin)      # Flow direction
    @variable(m, H_loss[1:K] >= 0)        # Head loss
    @variable(m, V_max >= V[1:K] >= 0)             # Flow velocity (m/s)

    # Flow directionality
    flow_dir = ones(K)

    if friction == "DW"
        @variable(m, relRough[1:K] >= 0)    # Relative pipe roughness
        @variable(m, Re[1:K] >= 0)          # Reynolds number
        @variable(m, 1 >= f[1:K] >= 0)           # Friction factor
    end

    @objective(m, Min, sum(pipeCost))

    for i = 1:N
        @constraint(m, source[i] - sink[i] + 
            sum(flow[pipe_index] for (pipe_index, node) in topology_dict if node[2] == i) - 
            sum(flow[pipe_index] for (pipe_index, node) in topology_dict if node[1] == i) >= 0)
    end

    for (pipe_index, node) in topology_dict
        @constraint(m, H_loss[pipe_index] <= H[node[1]] - H[node[2]]) # TODO: add directionality
    end

    # Now for the nonlinear constraints
    if oct
        gm = GlobalModel(model = m, name = "Hanoi")
        @variable(gm.model, Ls[1:K])
        set_lower_bound.(Ls, L)
        set_upper_bound.(Ls, L)
        for i = 1:K
            add_nonlinear_constraint(gm, :((D,Ls) -> 1.1 * D ^ 1.5 * Ls), vars = [D[i], Ls[i]], dependent_var = pipeCost[i])
            add_nonlinear_constraint(gm, :((flow, D) -> 4 * flow / (pi*D^2)), vars = [flow[i], D[i]], dependent_var = V[i], equality = true)
        end

        for pipe_index = 1:K
            if friction == "HW"
                add_nonlinear_constraint(gm, :((flow, D, Ls) -> $(flow_dir[pipe_index]) * 10.67 * Ls * flow ^ 1.852 /
                ($(rough[pipe_index])^1.852*D^4.8704)), 
                vars = [flow[pipe_index], D[pipe_index], Ls[pipe_index]], dependent_var = H_loss[pipe_index])
            elseif friction == "DW"
                throw(OCTHaGOnException("DW water network formulation incomplete. "))
                # @NLconstraint(m, H_loss[pipe_index] == f[pipe_index] * L[pipe_index] * V[pipe_index] ^ 2 / (
                #                         2 * D[pipe_index] * g))
                # @NLconstraint(m, relRough[pipe_index] == rough[pipe_index] / D[pipe_index])
                # @NLconstraint(m, Re[pipe_index] == rho * V[pipe_index] * D[pipe_index] / mu)
                # # From frictionFactorFitting.py
                # @NLconstraint(m, f[pipe_index] ^ 2.39794 >= 3.26853e-06 * Re[pipe_index] ^ 0.0574443 *
                #             relRough[pipe_index] ^ 0.364794 + 0.0001773 * Re[pipe_index] ^ -0.529499 *
                #             relRough[pipe_index] ^ -0.0810121 + 0.00301918 * Re[pipe_index] ^ -0.0220498 *
                #             relRough[pipe_index] ^ 1.73526 + 0.0734922 * Re[pipe_index] ^ -1.13629 *
                #             relRough[pipe_index] ^ 0.0574655 + 0.000214297 * Re[pipe_index] ^ 0.00035242 *
                #             relRough[pipe_index] ^ 0.823896)
            end
        end
        return gm
    else
        for i = 1:K
            @NLconstraint(m, pipeCost[i] >= 1.1*D[i]^1.5*L[i])
            @NLconstraint(m, V[i] == 4 * flow[i] / (pi*D[i]^2))
        end

        for pipe_index = 1:K
            if friction == "HW"
                @NLconstraint(m, H_loss[pipe_index] == flow_dir[pipe_index] * 10.67 * L[pipe_index] * flow[pipe_index] ^ 1.852 /
                                                    (rough[pipe_index]^1.852*D[pipe_index]^4.8704))
                            # S (hydraulic slope H/L) = 10.67*Q^1.852 (volumetric flow rate) /
                            # C^1.852 (pipe roughness) /d^4.8704 (pipe diameter)
            elseif friction == "DW"
                @NLconstraint(m, H_loss[pipe_index] == f[pipe_index] * L[pipe_index] * V[pipe_index] ^ 2 / (
                                        2 * D[pipe_index] * g))
                @NLconstraint(m, relRough[pipe_index] == rough[pipe_index] / D[pipe_index])
                @NLconstraint(m, Re[pipe_index] == rho * V[pipe_index] * D[pipe_index] / mu)
                # From frictionFactorFitting.py
                @NLconstraint(m, f[pipe_index] ^ 2.39794 >= 3.26853e-06 * Re[pipe_index] ^ 0.0574443 *
                            relRough[pipe_index] ^ 0.364794 + 0.0001773 * Re[pipe_index] ^ -0.529499 *
                            relRough[pipe_index] ^ -0.0810121 + 0.00301918 * Re[pipe_index] ^ -0.0220498 *
                            relRough[pipe_index] ^ 1.73526 + 0.0734922 * Re[pipe_index] ^ -1.13629 *
                            relRough[pipe_index] ^ 0.0574655 + 0.000214297 * Re[pipe_index] ^ 0.00035242 *
                            relRough[pipe_index] ^ 0.823896)
            end
        end
        return m
    end
end

# # Plotting results
# using LightGraphs
# using GraphPlot

# G = Graph(N)
# for i = 1:K
#     add_edge!(G, topology_dict[1][1], topology_dict[1][2])
# end

# gplot(G, nodelabel = 1:N)

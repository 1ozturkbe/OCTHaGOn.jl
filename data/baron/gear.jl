//  This is a gear train design problem taken from the GAMS test library
//  
//  A compound gear train is to be designed to achieve a specific
//  gear ratio between the driver and driven shafts. The objective
//  of the gear train design is to find the number of teeth of the
//  four gears and to obtain a required gear ratio of 1/6.931.
//  
//  The problem originated from:
//  Deb, K, and Goyal, M, Optimizing Engineering Designs Using a
//  Combined Genetic Search. In Back, T, Ed, Proceedings of the
//  Seventh International Conference on Genetic Algorithms. 1997,
//  pp. 521-528.

function gear(gm::Bool = false)
    m = JuMP.Model()
    @variable(m, 12 <= i[1:4] <= 60, Int)
    for j = 1:4
        JuMP.set_start_value(i[j], 24)
    end
    @constraint(m, - i[3] + i[4] >= 0)
    @constraint(m, i[1] - i[2] >= 0)
    if !gm
        @NLobjective(m, Min, (6.931 - i[1]*i[2]/(i[3]*i[4]))^2 + 1)
    else
        @variable(m, obj)
        gm = GlobalModel(m)
        add_nonlinear_constraint(gm, :((i, obj) -> obj - ((6.931 - i[1]*i[2]/(i[3]*i[4]))^2 + 1)))
        bound!(gm, obj => [0,2]) # Optimal objective is 1.
        return gm
    end
    return m
end
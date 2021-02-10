""" A small pooling problem. """
function pool1(gm::Bool = false)
    m = JuMP.Model(with_optimizer(CPLEX_SILENT))
    @variable(m, 0 <= x[1:7])
    lbs = Dict(x[2] => 3, x[3] => 1, x[4] => 2)
    [JuMP.set_lower_bound(key, value) for (key, value) in lbs]
    ubs = Dict(x[1] => 10, x[2] => 20, x[3] => 2, x[4] => 4, x[5] => 10, x[6] => 201, x[7] => 100)
    [JuMP.set_upper_bound(key, value) for (key, value) in ubs]
    @objective(m, Min, 6*x[1] + 16*x[2] -9*x[3] - 10*x[4])
    eqs = Dict("e1" => 12 - x[3]^2+x[4]^2, 
           "e2_1" => x[1]^2-x[2]^2+x[4]^2 - 3, "e2_2" => 100 - x[1]^2-x[2]^2+x[4]^2,
           "e3_1" => x[1]+x[2]-5*x[3]+2*x[4] - 10, "e3_2" => 20 - x[1]+x[2]-5*x[3]+2*x[4],
           "e4" => -x[4] + x[6]^2 - 4*x[7]^2)
    if gm
        gm = GlobalModel(model = m, name = "pool1")
        for (key, eq) in eqs
            add_nonlinear_constraint(gm, @constraint(gm.model, eq >= 0), name = key)
        end
        return gm
    else
        for (key, eq) in eqs
            @constraint(m, eq >= 0)
        end
        set_optimizer(m, BARON_SILENT)
        return m
    end
end


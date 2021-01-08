function nlp3(gm::Bool = false)
    m = JuMP.Model()
    @variable(m, 0 <= x[1:10])
    for i in [1, 2, 4, 6, 7, 8, 9, 10]
        JuMP.set_lower_bound(x[i], [1,1,1,85,90,3,1.2,145][i])
    end
    for i = 1:10
        JuMP.set_upper_bound(x[i], [2000, 16000, 120, 5000, 2000,
                                    93, 95, 12, 4, 162][i])
        JuMP.set_start_value(x[i], [1724.90452208, 16000,
                                    98.0900813608, 3049.1211364,
                                    1995.02326433, 90.718089075,
                                    94.2274481766, 10.432474977,
                                    2.59051951438, 149.682344530][i])
    end
    if !gm
        set_optimizer(m, BARON_SILENT)
        @NLconstraint(m, e1, x[1] - 1.22*x[4] + x[5] == 0)
        @NLconstraint(m, e2, x[9] + 0.222*x[10] == 35.82)    
        @NLconstraint(m, e3, 3*x[7] - x[10] == 133)    
        @NLconstraint(m, e4, x7 - 1.098*x8 + 0.038*(x8^2) - 0.325*(x6 - 89) == 86.35)    
        @NLconstraint(m, e5, x[4]*x[9]*x[6] + 1000*x[3]*x[6] - 98000*x[3] == 0)    
        @NLconstraint(m, e6, x[2] + x[5] - x[1]*x[8] == 0)    
        @NLconstraint(m, e7, 1.12*x[1] + 0.13167*x[8]*x[1] - 0.00667*(x[8]^2)*x[1] - x[4] >= 0)    
        @NLobj(m, Min, 5.04*x[1] + 0.035*x[2] + 10*x[3] + 3.36*x[5] - 0.063*x[4]*x[7])        
    else
        @variable(m, obj)
        @objective(m, Min, obj)
        gm = GlobalModel(model = m, name = "nlp3")
        set_optimizer(gm, CPLEX_SILENT)
        add_nonlinear_constraint(gm, :(x -> x[1] - 1.22*x[4] + x[5]), 
                                vars = [x[1], x[4], x[5]], name = "e1", equality = true)
        add_nonlinear_constraint(gm, :(x[9] + 0.222*x[10] - 35.82), 
                                vars = [x[9], x[10]], name = "e2", equality = true)
        add_nonlinear_constraint(gm, :(x -> 3*x[7] - x[10] - 133), 
                                vars = [x[7], x[10]], name = "e3", equality = true)
        add_nonlinear_constraint(gm, :(x -> x[7] - 1.098*x[8] + 0.038*(x[8]^2) - 0.325*(x[6] - 89) - 86.35), 
                                vars = [x[6], x[7], x[8]], name = "e4", equality = true)
        add_nonlinear_constraint(gm, :(x -> x[4]*x[9]*x[6] + 1000*x[3]*x[6] - 98000*x[3]), 
                                vars = [x[3], x[4], x[6], x[9]], name = "e5", equality = true)
        add_nonlinear_constraint(gm, :(x -> x[2] + x[5] - x[1]*x[8]), 
                                vars = [x[1], x[2], x[5], x[8]], name = "e6", equality = true)
        add_nonlinear_constraint(gm, :(x -> 1.12*x[1] + 0.13167*x[8]*x[1] - 0.00667*(x[8]^2)*x[1] - x[4]), 
                                vars = [x[1], x[4], x[8]], name = "e7")
        add_nonlinear_constraint(gm, :(x -> 5.04*x[1] + 0.035*x[2] + 10*x[3] + 3.36*x[5] - 0.063*x[4]*x[7]), 
                                vars = [x[1], x[2], x[3], x[4], x[5], x[7]], 
                                dependent_var = obj, name = "obj")
    end
    return m
end


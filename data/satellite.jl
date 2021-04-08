#=
Satellite optimization problem from from [Norheim, 2020]
Known optimum is 3.15 kg mass. 
=#

function satellite(solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))

    # Available components
    m[:D_P] = [0.025, 0.075, 0.1]
    m[:m_P] = [0.150, 1.75, 3.0]

    @variable(m, x[1:8])
    @objective(m, Min, x[8])
    gm = GlobalModel(model = m, name = "satellite")

    return gm
end
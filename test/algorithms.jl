#=
algorithms:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-27
=#

function test_sagemark3_find_bounds()
    gm = sagemark_to_GlobalModel(3; lse=false)
    set_optimizer(gm, Gurobi.Optimizer)
    old_bounds = get_bounds(gm)
    @test true
end
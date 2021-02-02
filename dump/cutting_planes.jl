""" Solves the QP with cutting planes only. """
function cutting_plane_solve(gm::GlobalModel)
    bbr = gm.bbls[1]
    gradvals = evaluate_gradient(bbr, bbr.X)
    # Testing adding gradient cuts
    for i=1:size(bbr.X, 1)
        @constraint(gm.model, bbr.dependent_var >= sum(gradvals[i] .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i])
    end
    optimize!(gm)   
end
# Only one feasible leaf (4), so can just use regression equations
# Regressions over leaves
P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [outputs[idx] for idx in idxs]
FOMs = [P_shaft, Rotational_Speed, Mass, Efficiency]
leaf_index = IAI.apply(lnr, log.(X_feas))
for FOM in FOMs
    regressor = regress(log.(X_feas), log.(Y_feas[!,string(FOM)]))
    constant = IAI.get_prediction_constant(regressor)
    weights  = IAI.get_prediction_weights(regressor)[1]
    vks = Symbol.(names(X_feas))
    β = []
    for i = 1:size(vks, 1)
        if vks[i] in keys(weights)
            append!(β, weights[vks[i]])
        else
            append!(β, 0.0)
        end
    end
    @constraint(m, sum(β .* inputs) + constant == FOM)
end

# Solving
@constraint(m, log(9.8) <= P_shaft)
@constraint(m, log.(7800) <= Rotational_Speed <= log(8200))
@objective(m, Min, Mass)
optimize!(m)
println("Inputs")
for i=1:length(inputs)
    println(string(inputs[i], " ", exp(getvalue(inputs[i]))))
end
println("FOMs")
for FOM in FOMs
    println(string(FOM, " ", exp(getvalue(FOM))))
end

# fdf = DataFrame(names(X) .=> exp.(getvalue.(inputs)))
# CSV.write("data/afpm/afpm_opt.csv", fdf)
""" Creates and tests the axial flux motor model. """
function afpm_model()
    # Data preprocessing
    X = log.(CSV.read("data/afpm/afpm_inputs.csv", DataFrame,
                 copycols=true, delim=","));
    X_infeas = log.(CSV.read("data/afpm/afpm_infeas_inputs.csv", DataFrame,
                        copycols=true, delim=","));
    X = select!(X, Not(:Column1));
    X_infeas = select!(X_infeas, Not(:Column1));

    Y = CSV.read("data/afpm/afpm_outputs.csv", DataFrame, copycols=true, delim=","); # log taken later
    foms = ["P_shaft", "Torque", "Rotational Speed", "Efficiency", "Mass", "Mass Specific Power"]
    Y = select!(Y, foms);

    feas_idxs = findall(x -> !ismissing(x) && x .>= 0.5, Y.Efficiency);
    X_feas = X[feas_idxs, :]
    Y_feas = log.(Y[feas_idxs, :])
 
    # Creating model with relevant variables, and geometry constraints
    m = Model(with_optimizer(OCTHaGOn.SOLVER_SILENT))
    add_variables_from_data!(m, X)
    bound_to_data!(m, X_feas)
    add_variables_from_data!(m, Y)
    bound_to_data!(m, Y_feas)
    @assert length(all_variables(m)) == 15
    N_coils, TPC, p = m[:N_coils], m[:TPC], m[:p]

    # Geometry constraints (in logspace)
    D_out, D_in, N_coils, wire_w, wire_A, wire_h = m[:D_out], m[:D_in], m[:N_coils], 
                                                   m[:wire_w], m[:wire_A], m[:wire_h]
    @constraint(m, D_out >= D_in)
    @constraint(m, log(pi) + D_in >= log(0.2) + wire_w + N_coils) #pi*D_in >= 2*0.1*wire_w*N_coils
    @constraint(m, wire_A == wire_h + wire_w)
    @constraint(m, N_coils >= p + 1e-3) # motor type 2

    # Objectives and FOMs
    P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [m[Symbol(fom)] for fom in foms]
    gm = GlobalModel(model = m)

    # Integer constraints (in logspace)
    restrict_to_set(N_coils, unique(X_feas[!, "N_coils"]));
    restrict_to_set(p, unique(X_feas[!, "p"]))
    restrict_to_set(TPC, unique(X_feas[!, "TPC"]))

    # Feasibility constraint
    add_datadriven_constraint(gm, X, Int64.([!ismissing(y) && y .>= 0.5 for y in Array(Y.Efficiency)]) .* 2 .- 1)
    # FOM regressions
    for fom in foms
        add_datadriven_constraint(gm, X_feas, Array(Y_feas[!, Symbol(fom)]), dependent_var = m[Symbol(fom)], equality=true)
    end

    # Requirements and objective
    @constraint(m, log(7900) <= Rotational_Speed <= log(8100))
    @constraint(m, log(9.8) <= P_shaft <= log(10.2))
    @objective(m, Min, Mass)

    return gm
end



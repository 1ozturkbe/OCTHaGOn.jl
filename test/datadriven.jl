using DataFrames, CSV

"""
    add_variables_from_data!(gm::Union{JuMP.Model, GlobalModel},
                            X::DataFrame)

Adds/finds variables depending on the columns of X. 
"""
function add_variables_from_data!(m::JuMP.Model,
                                 X::DataFrame)
    varkeys = names(X)
    vars = []
    data_vars = []
    for key in varkeys
        try 
            push!(vars, m[Symbol(key)])
        catch KeyError
            nv = @variable(m, base_name = string(key))
            m[Symbol(key)] = nv
            push!(vars, nv)
            push!(data_vars, Symbol(key))
        end
    end
    count != 0 && @info("Added $(length(data_vars)) variables from data: " * string(data_vars))
    return
end

add_variables_from_data(gm::GlobalModel, X) = add_variables_from_data(gm.model, X)


""" 
    bound_to_data(gm::Union{JuMP.Model, GlobalModel},
                  X::DataFrame)

Constrains the domain of relevant variables to the box interval defined by X.
"""
function bound_to_data!(m::JuMP.Model,
                       X::DataFrame)
    vars = []
    try 
        vars = [m[Symbol(key)] for key in names(X)]
    catch
        throw(OCTException("Please make sure the keys of X match with variables in model $(gm)."))
    end
    ranges = Dict(m[Symbol(key)] => [minimum((X[!, Symbol(key)])), maximum((X[!, Symbol(key)]))] for key in names(X))
    bound!(m, ranges);
    return
end
    
bound_to_data!(gm::GlobalModel, X::DataFrame) = bound_to_data(gm.model, X)

"""
    restrict_to_set(var::JuMP.VariableRef, s::Union{Set, Array})

Restricts variable to a set s. Useful for non-integer sets or when taking log of integer variables. 
"""
function restrict_to_set(var::JuMP.VariableRef, s::Union{Set, Array})
    int = @variable(var.model, [1:length(s)], Bin)
    @constraint(var.model, sum(int) == 1)
    @constraint(var.model, var == sum(s .* int))
    return
end

# """
#  Adds a data-driven constraint to GlobalModel. Data driven BBLs
# do not allow of resampling. 
# """
# function add_datadriven_constraint(gm::GlobalModel,
#                      X::DataFrame, Y::Union{Array, DataFrame};
#                      vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
#                      dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
#                      name::String = "bbl" * string(length(gm.bbls) + 1),
#                      equality::Bool = false)
#     if isnothing(vars)

    
#     if isnothing(dependent_var)
#         new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
#                                         equality = equality, name = name)
#         push!(gm.bbls, new_bbl)
#         return
#     else
#         new_bbl = BlackBoxRegressor(constraint = constraint, vars = vars, expr_vars = expr_vars,
#                                     dependent_var = dependent_var, equality = equality, name = name)
#         push!(gm.bbls, new_bbl)
#         return
#     end
# end

""" Creates an axial flux motor model. """
function test_afpm()
    m = Model(with_optimizer(CPLEX_SILENT))

    # Data preprocessing
    X = log.(CSV.read("data/afpm/afpm_inputs.csv", DataFrame,
                 copycols=true, delim=","));
    X_infeas = log.(CSV.read("data/afpm/afpm_infeas_inputs.csv", DataFrame,
                        copycols=true, delim=","));
    X = select!(X, Not(:Column1));
    X_infeas = select!(X_infeas, Not(:Column1));

    Y = CSV.read("data/afpm/afpm_outputs.csv", DataFrame, copycols=true, delim=","); # log taken later
    Y = select!(Y, Not(:Column1));

    feas_idxs = findall(x -> !ismissing(x) && x .>= 0.5, Y.Efficiency);
    X_feas = X[feas_idxs, :]
    Y_feas = Y[feas_idxs, :]
 
    # Creating model with relevant variables, and geometry constraints
    m = Model(with_optimizer(CPLEX_SILENT))
    add_variables_from_data!(m, X)
    add_variables_from_data!(m, Y)
    @test length(all_variables(m)) == 26
    N_coils, TPC, p = m[:N_coils], m[:TPC], m[:p]

    [JuMP.set_integer(var) for var in [N_coils, TPC, p]]

    # Geometry constraints (in logspace)
    D_out, D_in, N_coils, wire_w = m[:D_out], m[:D_in], m[:N_coils], m[:wire_w]
    @constraint(m, D_out >= D_in)
    @constraint(m, log(pi) + D_in >= log(0.2) + wire_w + N_coils) #pi*D_in >= 2*0.1*wire_w*N_coils
    @constraint(m, N_coils >= p + 1e-3) # motor type 2

    # Integer constraints (in logspace)
    restrict_to_set(N_coils, unique(X_feas[!, "N_coils"]));
    restrict_to_set(p, unique(X_feas[!, "p"]))
    restrict_to_set(TPC, unique(X_feas[!, "TPC"]))

    # Objectives and FOMs
    foms = ["P_shaft", "Torque", "Rotational Speed", "Efficiency", "Mass", "Mass Specific Power"]
    P_shaft, Torque, Rotational_Speed, Efficiency, Mass, Mass_Specific_Power = [m[Symbol(fom)] for fom in foms]
    set_upper_bound(Efficiency, 1)
    
    @test true
end


test_afpm()


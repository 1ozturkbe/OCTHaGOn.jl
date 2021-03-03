using DataFrames, CSV

"""
    add_variables_from_data(gm::Union{JuMP.Model, GlobalModel},
                            X::DataFrame)

Adds/finds variables depending on the columns of X. 
"""
function add_variables_from_data(gm::Union{JuMP.Model, GlobalModel},
                                 X::DataFrame)
    varkeys = names(X)
    m = gm
    if gm isa GlobalModel
        m = gm.model
    end
    vars = []
    data_vars = []
    for key in in varkeys
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

"""

 Adds a data-driven constraint to GlobalModel. Data driven BBLs
do not allow of resampling. 
"""
function add_datadriven_constraint(gm::GlobalModel,
                     X::DataFrame, Y::Union{Array, DataFrame};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     dependent_var::Union{Nothing, JuMP.VariableRef} = nothing,
                     name::String = "bbl" * string(length(gm.bbls) + 1),
                     equality::Bool = false)
    if isnothing(vars)

    
    if isnothing(dependent_var)
        new_bbl = BlackBoxClassifier(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                        equality = equality, name = name)
        push!(gm.bbls, new_bbl)
        return
    else
        new_bbl = BlackBoxRegressor(constraint = constraint, vars = vars, expr_vars = expr_vars,
                                    dependent_var = dependent_var, equality = equality, name = name)
        push!(gm.bbls, new_bbl)
        return
    end
end

""" Creates an axial flux motor model. """
function test_afpm()
    m = Model(with_optimizer(CPLEX_SILENT))

    using DataFrames, CSV

    # Data preprocessing
    X = CSV.read("data/afpm/afpm_inputs.csv", DataFrame,
                 copycols=true, delim=",");
    X_infeas = CSV.read("data/afpm/afpm_infeas_inputs.csv", DataFrame,
                        copycols=true, delim=",");
    Y = CSV.read("data/afpm/afpm_outputs.csv", DataFrame,
                    copycols=true, delim=",");
    X = select!(X, Not(:Column1));
    X_infeas = select!(X_infeas, Not(:Column1));
    Y = select!(Y, Not(:Column1));

    # Creating model with relevant variables, and geometry constraints
    m = Model(with_optimizer(CPLEX_SILENT))
    add_variables_from_data(m, X)
    add_variables_from_data(m, Y)
    @test all_variables(m) == 26

end

test_afpm()


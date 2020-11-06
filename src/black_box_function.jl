#=
sample_data:
- Julia version: 
- Author: Berk
- Date: 2020-07-26
=#

"""
    get_varmap(vars::Array)

Helper function to map vars to flatvars.
Arguments:
    flatvars is a flattened Array{JuMP.VariableRef}
    vars is the unflattened version
Returns:
    Dict of ID maps
"""

function get_varmap(vars::Array)
    idxs = []
    count = 0
    for i = 1:length(vars)
        count = 1
        if vars[i] isa JuMP.VariableRef
            push!(idxs, i)
        else
            append!(idxs, [(i,j) for j=1:length(vars[i])])
        end
    end
    return idxs
end

function infarray(varmap::Array)
    arr = []
    for idx in varmap
        if idx isa Tuple && idx[2] == 1
            push!(arr, [Inf])
        elseif idx isa Integer
            push!(arr, Inf)
        else
            push!(arr[end], Inf)
        end
    end
    return arr
end

function deconstruct(data::DataFrame, vars::Array, varmap::Array)
    n_samples, n_vars = size(data)
    infarr = infarray(varmap)
    arrs = [];
    stringvars = string.(collect(Iterators.flatten(vars)))
    for i = 1:n_samples
        narr = deepcopy(infarr)
        for j = 1:length(varmap)
            if varmap[j] isa Tuple
                narr[varmap[j][1]][varmap[j][2]] = data[i, stringvars[j]]
            else
                narr[varmap[j]] = data[i, stringvars[j]]
            end
        end
        push!(arrs, narr)
    end
    return arrs
end

@with_kw mutable struct BlackBoxFunction
"""
Contains all required info to be able to generate a global optimization constraint.
"""
    constraint::Union{JuMP.ConstraintRef, Expr}        # The "raw" constraint
    vars::Array                                        # JuMP variables
    name::Union{String, Real} = ""                     # Function name
    fn::Union{Nothing, Function} = functionify(constraint)   # ... and actually evaluated f'n
    varmap::Array = get_varmap(vars)                   # ... with the required varmapping.
    X::DataFrame = DataFrame([Float64 for i=1:length(varmap)], string.(collect(Iterators.flatten(vars))))
                                                       # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    threshold_accuracy::Float64 = 0.95                 # Minimum tree accuracy
    threshold_feasibility::Float64 = 0.15              # Minimum feas_ratio
    n_samples::Int = 100                               # For next set of samples, set and forget.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    tags::Array{String} = []                           # Other tags
end

"""
    evaluate(constraint::JuMP.ConstraintRef, data::Union{Dict, DataFrame})

Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform.
"""

function evaluate(bbf::BlackBoxFunction, data::Union{Dict, DataFrame})
    clean_data = data_to_DataFrame(data);
    if isnothing(bbf.fn)
        constr_obj = constraint_object(bbf.constraint)
        rhs_const = get_constant(constr_obj.set)
        vals = [JuMP.value(constr_obj.func, i -> get(clean_data, string(i), Inf)[j]) for j=1:size(clean_data,1)]
        if any(isinf.(vals))
            throw(OCTException(string("Constraint ", bbf.constraint, " returned an infinite value.")))
        end
        if isnothing(rhs_const)
            vals = [-1*distance_to_set(vals[i], constr_obj.set) for i=1:length(vals)]
        else
            coeff=1
            if constr_obj.set isa MOI.LessThan
                coeff=-1
            end
            vals = coeff.*(vals .- rhs_const)
        end
        length(vals) == 1 && return vals[1]
        return vals
    else
        arrs = deconstruct(clean_data, bbf.vars, bbf.varmap)
        vals = [bbf.fn(arr...) for arr in arrs]
        length(vals) == 1 && return vals[1]
        return vals    end
end

"""
    linearize_objective!(model::JuMP.Model)
Makes sure that the objective function is affine.
"""
function linearize_objective!(model::JuMP.Model)
    objtype = JuMP.objective_function(model)
    objsense = string(JuMP.objective_sense(model))
    if objtype isa Union{VariableRef, GenericAffExpr} || objsense == "FEASIBILITY_SENSE"
        return
    else
        aux = @variable(model)
        @objective(model, Min, aux)
        # Default optimization problem is always a minimization
        coeff = 1;
        objsense == "MAX_SENSE" && (coeff = -1)
        try
            @constraint(model, aux >= coeff*JuMP.objective_function(model))
        catch
            @NLconstraint(model, aux >= coeff*JuMP.objective_function(model))
        end
        return
    end
end

function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbf, x)
end

function eval!(bbf::BlackBoxFunction, X::DataFrame)
    """ Evaluates BlackBoxFunction at all X and stores the resulting data. """
    values = bbf(X);
    append!(bbf.X, X[:,string.(bbf.vars)], cols=:setequal)
    append!(bbf.Y, values);
    bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    return
end



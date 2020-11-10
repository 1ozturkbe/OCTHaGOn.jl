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
    vars is the unflattened version, usually derived from an Expr.
Returns:
    Dict of ID maps
"""
function get_varmap(expr_vars::Array, vars::Array)
    length(flat(expr_vars)) >= length(vars) || throw(OCTException(string("Insufficiently many input
    variables declared in ", vars, ".")))
    unique(vars) == vars || throw(OCTException(string("Nonunique variables among ", vars, ".")))
    varmap = Tuple[(0,0) for i=1:length(vars)]
    for i = 1:length(expr_vars)
        if expr_vars[i] isa JuMP.VariableRef
            try
                varmap[findall(x -> x == expr_vars[i], vars)[1]] = (i,0)
            catch
                throw(OCTException(string("Scalar variable ", expr_vars[i], " was not properly declared in vars.")))
            end
        else
            for j=1:length(expr_vars[i])
                try
                    varmap[findall(x -> x == expr_vars[i][j], vars)[1]] = (i,j)
                catch
                    continue
                end
            end
        end
    end
    return varmap
end


"""
    infarray(varmap::Array)

Creates a template array for deconstruct function.
"""
function infarray(varmap::Array)
    arr = []
    # Pre allocate and fill!
    copied_map = sort(deepcopy(varmap))
    firsts = first.(copied_map)
    seconds = last.(copied_map)
    arr_length = length(unique(firsts))
    for i=1:arr_length
        tup_idx = findall(j -> j[1] == i, copied_map)
        if seconds[tup_idx] == [0]
            push!(arr, Inf)
        else
            push!(arr, Inf*ones(maximum(seconds[tup_idx])))
        end
    end
    return arr
end

"""
    deconstruct(data::DataFrame, vars::Array, varmap::Array)

Takes in data for input into a Function, and rips it apart into appropriate arrays.
"""
function deconstruct(data::DataFrame, vars::Array, varmap::Array)
    n_samples, n_vars = size(data)
    infarr = infarray(varmap)
    arrs = [];
    stringvars = string.(vars)
    for i = 1:n_samples
        narr = deepcopy(infarr)
        for j = 1:length(varmap)
            if varmap[j] isa Tuple && varmap[j][2] != 0
                narr[varmap[j][1]][varmap[j][2]] = data[i, stringvars[j]]
            else
                narr[varmap[j][1]] = data[i, stringvars[j]]
            end
        end
        push!(arrs, narr)
    end
    return arrs
end

"""
    @with_kw mutable struct BlackBoxFunction

Contains all required info to be able to generate a global optimization constraint.
Mandatory arguments:
    constraint::Union{JuMP.ConstraintRef, Expr}
    vars::Array{JuMP.VariableRef,1}

Also contains data w.r.t. samples from the function.
Can be tagged with additional info.
"""
@with_kw mutable struct BlackBoxFunction
    constraint::Union{JuMP.ConstraintRef, Expr}        # The "raw" constraint
    vars::Array{JuMP.VariableRef,1}                      # JuMP variables (flat)
    name::Union{String, Real} = ""                     # Function name
    fn::Union{Nothing, Function} = functionify(constraint)   # ... and actually evaluated f'n
    expr_vars = vars_from_expr(constraint, vars)    # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    X::DataFrame = DataFrame([Float64 for i=1:length(varmap)], string.(vars))
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
        if any(isinf.(vals)) || any(isnan.(vals))
            throw(OCTException(string("Constraint ", bbf.constraint, " returned an infinite or NaN value.",
                                      "Please check your variable definitions in BBF ", bbf, " . ")))
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
    function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})

Makes the BBF callable as a function.
"""
function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbf, x)
end

""" Evaluates BlackBoxFunction at all X and stores the resulting data. """
function eval!(bbf::BlackBoxFunction, X::DataFrame)
    values = bbf(X);
    append!(bbf.X, X[:,string.(bbf.vars)], cols=:setequal)
    append!(bbf.Y, values);
    bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    return
end



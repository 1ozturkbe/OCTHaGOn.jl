"""
    @with_kw mutable struct BlackBoxRegressor

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbfs using functions:
    add_nonlinear_constraints
    add_nonlinear_or_compatible

Mandatory arguments:
    constraint::Union{JuMP.ConstraintRef, Expr}
        A function
    vars::Array{JuMP.VariableRef,1}, 
    dependent_var::JuMP.VariableRef

Optional arguments:
    expr_vars::Union{Array, Nothing}
        JuMP variables as function arguments (i.e. vars rolled up into vector forms).
        vars ⋐ flat(expr_vars)
    name::String
    equality::Bool
        Specifies whether function should be satisfied to an equality
"""
@with_kw mutable struct BlackBoxRegressor
    constraint::Union{JuMP.ConstraintRef, Expr}        # The "raw" constraint
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    dependent_var::JuMP.VariableRef                    # Dependent variable
    name::String = ""                                  # Function name
    expr_vars:: Union{Array, Nothing} = nothing        # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    fn::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars)) # Function samples
    Y::Array = []                                      # Function values
    feas_idxs::Array = []                              # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.GridSearch} = []               # Learners...
    learner_kwargs = []                                # and their kwargs... 
    feas_learners::Array{IAI.GridSearch} = []          # Classification learners...
    feas_learner_kwargs = []                           # and their kwargs...
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbr_defaults()                      # Relevant settings
end

function Base.show(io::IO, bbc::BlackBoxRegressor)
    println(io, "BlackBoxClassifier " * bbc.name * " with $(length(bbc.vars)) dependent variables: ")
    println(io, "Sampled $(length(bbc.Y)) times, and has $(length(bbc.learners)) trained ORTs.")
end

"""
    @with_kw mutable struct BlackBoxClassifier

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbfs using functions:
    add_nonlinear_constraints
    add_nonlinear_or_compatible

Mandatory arguments:
    constraint::Union{JuMP.ConstraintRef, Expr}
        A function
    vars::Array{JuMP.VariableRef,1}

Optional arguments:
    expr_vars::Union{Array, Nothing}
        JuMP variables as function arguments (i.e. vars rolled up into vector forms).
        vars ⋐ flat(expr_vars)
    name::String
    equality::Bool
        Specifies whether function should be satisfied to an equality
"""
@with_kw mutable struct BlackBoxClassifier
    constraint::Union{JuMP.ConstraintRef, Expr}        # The "raw" constraint
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    name::String = ""                                  # Function name
    expr_vars:: Union{Array, Nothing} = nothing        # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    fn::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars))
                                                       # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    dependent_var::Union{JuMP.VariableRef, Nothing} = nothing
    learners::Array{IAI.GridSearch} = []               # Learners...
    # learner_data::Array{LearnerData} = []              # Constraints training data
    learner_kwargs = []                                # And their kwargs... 
    mi_constraints::Array = []                         # and their corresponding MI constraints,
    leaf_variables::Array = []                         # and their binary leaf variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbc_defaults()                      # Relevant settings
end

function Base.show(io::IO, bbc::BlackBoxClassifier)
    println(io, "BlackBoxClassifier " * bbc.name * " with $(length(bbc.vars)) variables: ")
    println(io, "Sampled $(length(bbc.Y)) times, and has $(length(bbc.learners)) trained OCTs.")
end

""" BlackBoxFunction isa a supertype."""
BlackBoxFunction = Union{BlackBoxClassifier, BlackBoxRegressor}

set_param(bbf::BlackBoxFunction, key::Symbol, val) = set_param(bbf.params, key, val)
set_param(bbfs::Array{BlackBoxFunction}, key::Symbol, val) = foreach(bbf -> set_param(bbf.params, key, val), bbfs)
get_param(bbf::BlackBoxFunction, key::Symbol) = get_param(bbf.params, key)

"""
    add_data!(bbf::Union{BlackBoxFunction, DataConstraint}, X::DataFrame, Y::Array)

Adds data to a BlackBoxFunction or DataConstraint.
"""
function add_data!(bbf::Union{BlackBoxFunction, DataConstraint}, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    append!(bbf.X, X[:,string.(bbf.vars)], cols=:setequal)
    append!(bbf.Y, Y)
    bbf.feas_ratio = sum(bbf.Y .>= 0)/length(bbf.Y); #TODO: optimize.
    return
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
        vals = [JuMP.value(constr_obj.func, i -> clean_data[!,string(i)][j]) for j=1:size(clean_data,1)]
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
        vals = []
        for arr in arrs
            try
                push!(vals, Base.invokelatest(bbf.fn, (arr...)))
            catch DomainError
                push!(vals, -Inf)
            end
        end
        return vals
    end
end

"""
    function (bbf::BlackBoxFunction)(x::Union{DataFrame,Dict,DataFrameRow})

Makes the BlackBoxFunction callable as a function.
"""
function (bbf::BlackBoxFunction)(X::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbf, X)
end

""" Evaluates BlackBoxFunction at all X and stores the resulting data. """
function eval!(bbf::BlackBoxFunction, X::DataFrame)
    Y = bbf(X);
    add_data!(bbf, X, Y)
end

""" Deletes all data associated with a BlackBoxFunction or GlobalModel. """
function clear_data!(bbf::BlackBoxFunction)
    bbf.X = DataFrame([Float64 for i=1:length(bbf.vars)], string.(bbf.vars))
    bbf.Y = [];
    bbf.feas_ratio = 0
    bbf.learners =[];
    bbf.learner_kwargs = []                            
    bbf.accuracies = []                    # and the tree misclassification scores.
end
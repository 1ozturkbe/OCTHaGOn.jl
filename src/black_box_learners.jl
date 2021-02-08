"""
    @with_kw mutable struct BlackBoxRegressor

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbls using functions:
    add_nonlinear_constraints
    add_nonlinear_or_compatible

Mandatory arguments:
    constraint::Union{JuMP.ConstraintRef, Expr}
        A function
    vars::Array{JuMP.VariableRef,1}, 
    expr_vars::Array
    dependent_var::JuMP.VariableRef

Optional arguments:
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
    expr_vars::Array                                   # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    f::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    g::Union{Nothing, Function} = gradientify(constraint, expr_vars)   # ... and its gradient f'n
    X::DataFrame = DataFrame([Float64 
                 for i=1:length(vars)], string.(vars)) # Function samples
    Y::Array = []                                                           # Function values
    gradients::DataFrame = DataFrame([Union{Missing, Float64} 
                 for i=1:length(vars)], string.(vars)) # Gradients
    curvatures::Array = []                             # Curvature around the points
    infeas_X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars)) # Infeasible samples, if any
    equality::Bool = false                             # Equality check
    learners::Array{Union{IAI.OptimalTreeRegressor, IAI.OptimalTreeClassifier}} = []     # Learners...
    learner_kwargs = []                                # and their kwargs... 
    thresholds::Array{Union{Nothing, Pair}} = []       # For thresholding. 
    ul_data::Array{Dict} = Dict[]                      # Upper/lower bounding data
    active_trees::Dict{Int64, Union{Nothing, Pair}} = Dict() # Currently active tree indices
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}() # and their corresponding MI constraints,
    leaf_variables::Dict = Dict{Int64, JuMP.VariableRef}() # and their leaves and leaf variables
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbr_defaults(length(vars))          # Relevant settings
end

function Base.show(io::IO, bbr::BlackBoxRegressor)
    println(io, "BlackBoxRegressor " * bbr.name * " with $(length(bbr.vars)) independent variables: ")
    println(io, "Sampled $(length(bbr.Y)) times, and has $(length(bbr.learners)) trained ORTs.")
end

"""
    @with_kw mutable struct BlackBoxClassifier

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbls using functions:
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
    expr_vars::Array                                   # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    f::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    g::Union{Nothing, Function} = gradientify(constraint, expr_vars)   # ... and its gradient f'n
    X::DataFrame = DataFrame([Float64 for i=1:length(vars)], string.(vars))
                                                       # Function samples
    Y::Array = []                                      # Function values
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    equality::Bool = false                             # Equality check
    learners::Array{IAI.OptimalTreeClassifier} = []    # Learners...
    learner_kwargs = []                                # And their kwargs... 
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}() # and their corresponding MI constraints,
    leaf_variables::Dict = Dict{Int64, JuMP.VariableRef}() # and their binary leaves and associated variables,
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbc_defaults(length(vars))          # Relevant settings
end

function Base.show(io::IO, bbc::BlackBoxClassifier)
    println(io, "BlackBoxClassifier " * bbc.name * " with $(length(bbc.vars)) variables: ")
    println(io, "Sampled $(length(bbc.Y)) times, and has $(length(bbc.learners)) trained OCTs.")
end

""" BBL type is for function definitions! """
BlackBoxLearner = Union{BlackBoxClassifier, BlackBoxRegressor}

set_param(bbl::BlackBoxLearner, key::Symbol, val) = set_param(bbl.params, key, val)
set_param(bbls::Array{BlackBoxLearner}, key::Symbol, val) = foreach(bbl -> set_param(bbl, key, val), bbls)
get_param(bbl::BlackBoxLearner, key::Symbol) = get_param(bbl.params, key)
get_param(bbls::Array{BlackBoxLearner}, key::Symbol) = [get_param(bbl, key) for bbl in bbls]

"""
    add_data!(bbc::BlackBoxClassifier, X::DataFrame, Y::Array)

Adds data to BlackBoxClassifier.
"""
function add_data!(bbc::BlackBoxClassifier, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    if size(bbc.X,1) == 0
        bbc.feas_ratio = sum(Y .>= 0)/length(Y)
    else
        bbc.feas_ratio = (bbc.feas_ratio*size(bbc.X,1) + sum(Y .>= 0))/(size(bbc.X, 1) + length(Y))
    end    
    append!(bbc.X, X, cols=:intersect)
    append!(bbc.Y, Y)
    return
end

"""
    add_data!(bbr::BlackBoxRegressor, X::DataFrame, Y::Array)

Adds data to BlackBoxRegressor.
"""
function add_data!(bbr::BlackBoxRegressor, X::DataFrame, Y::Array)
    @assert length(Y) == size(X, 1)
    infeas_idxs = findall(x -> isinf(x), Y)
    if !isempty(infeas_idxs)
        append!(bbr.infeas_X, X[infeas_idxs, :], cols=:intersect)
        clean_X = delete!(X, infeas_idxs)
        append!(bbr.gradients, DataFrame(missings(size(clean_X, 1), 
                               length(bbr.vars)), string.(bbr.vars)), cols=:intersect)
        append!(bbr.curvatures, missings(size(clean_X,1)))
        append!(bbr.X, clean_X, cols=:intersect)
        append!(bbr.Y, deleteat!(Y, infeas_idxs))
    else
        append!(bbr.gradients, DataFrame(missings(size(X, 1), 
                               length(bbr.vars)), string.(bbr.vars)), cols=:intersect)
        append!(bbr.curvatures, missings(size(X,1)))
        append!(bbr.X, X, cols=:intersect)
        append!(bbr.Y, Y)
    end
    return
end

"""
    evaluate(bbl::BlackBoxLearner, data::Union{Dict, DataFrame})

Evaluates constraint violation on data in the variables, and returns distance from set.
        Note that the keys of the Dict have to be uniform.
"""
function evaluate(bbl::BlackBoxLearner, data::Union{Dict, DataFrame})
    clean_data = data_to_DataFrame(data);
    if isnothing(bbl.f)
        constr_obj = constraint_object(bbl.constraint)
        rhs_const = get_constant(constr_obj.set)
        vals = [JuMP.value(constr_obj.func, i -> clean_data[!,string(i)][j]) for j=1:size(clean_data,1)]
        if any(isinf.(vals)) || any(isnan.(vals))
            throw(OCTException(string("Constraint ", bbl.constraint, " returned an infinite or NaN value.",
                                      "Please check your variable definitions in bbl ", bbl, " . ")))
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
        arrs = deconstruct(clean_data, bbl.vars, bbl.expr_vars, bbl.varmap)
        vals = []
        for arr in arrs
            try
                push!(vals, Base.invokelatest(bbl.f, (arr...)))
            catch DomainError
                push!(vals, -Inf)
            end
        end
        return vals
    end
end

"""
    evaluate_gradient(bbl::BlackBoxLearner, data::Union{Array, DataFrame})

Evaluates gradient of function through ForwardDiff. 
TODO: speed-ups!. 
"""
function evaluate_gradient(bbl::BlackBoxLearner, data::DataFrame)
    @assert(size(data, 2) == length(bbl.vars))
    if bbl.constraint isa JuMP.ConstraintRef
        return DataFrame(hcat([Base.invokelatest(bbl.g, Array(row)) 
                for row in eachrow(data[:, string.(bbl.vars)])]...)', string.(bbl.vars))
    elseif bbl.constraint isa Expr
        arrs = deconstruct(data, bbl.vars, bbl.expr_vars, bbl.varmap)
        return DataFrame(hcat([Base.invokelatest(bbl.g, Float64[flat(arr)...]) 
                for arr in arrs]...)', string.(bbl.vars))
    end
end

"""
    update_gradients(bbl::BlackBoxLearner, idxs = collect(1:size(bbl.X,1)))

Updates gradient information of selected points. 
"""
function update_gradients(bbl::BlackBoxLearner, idxs = collect(1:size(bbl.X,1)))
    bbl.gradients[idxs, :] = evaluate_gradient(bbl, bbl.X[idxs, :])
    return
end    

"""
    function (bbl::BlackBoxLearner)(x::Union{DataFrame,Dict,DataFrameRow})

Makes the BBC or BBR callable as a function.
"""
function (bbl::BlackBoxLearner)(X::Union{DataFrame,Dict,DataFrameRow})
    return evaluate(bbl, X)
end

""" Evaluates BlackBoxLearner at all X and stores the resulting data. """
function eval!(bbl::BlackBoxLearner, X::DataFrame)
    Y = bbl(X);
    add_data!(bbl, X, Y)
end

""" Deletes all data associated with object. """
function clear_data!(bbc::BlackBoxClassifier)
    bbc.X = DataFrame([Float64 for i=1:length(bbc.vars)], string.(bbc.vars))
    bbc.Y = [];
    bbc.feas_ratio = 0
    bbc.learners = [];
    bbc.learner_kwargs = []                            
    bbc.accuracies = []
end

function clear_data!(bbr::BlackBoxRegressor)
    bbr.X = DataFrame([Float64 for i=1:length(bbr.vars)], string.(bbr.vars))
    bbr.Y = [];
    bbr.infeas_X = DataFrame([Float64 for i=1:length(bbr.vars)], string.(bbr.vars));
    bbr.learners = [];
    bbr.learner_kwargs = []   
    bbr.thresholds = []                         
    bbr.ul_data = Dict[]                          
end

""" Deletes tree data associated with object. """
function clear_tree_data!(bbc::BlackBoxClassifier)
    bbc.learners = [];
    bbc.learner_kwargs = []                            
    bbc.accuracies = []
end

function clear_tree_data!(bbr::BlackBoxRegressor)
    bbr.learners = [];
    bbr.learner_kwargs = []   
    bbr.thresholds = []                         
    bbr.ul_data = Dict[]            
end

""" 
    find_leaf_of_soln(bbl::BlackBoxLearner)

Find leaf (or leaves) of previous solution via binary variables. 
"""
function find_leaf_of_soln(bbc::BlackBoxClassifier)
    if !bbc.equality
        leaf_in = 0
        for (leaf, var) in bbc.leaf_variables
            if isapprox(getvalue(var), 1; atol=1e-5)
                leaf_in = leaf
                break
            end
        end
        leaf_in != 0 || throw(OCTException("BBC $(bbc.name) did not return a valid leaf. "*
                                           "Please make sure the MI constraints are added to a GM."))
        return leaf_in
    else
        leaf_in = []
        for (leaf, var) in bbc.leaf_variables
            if isapprox(getvalue(var), 1; atol=1e-5)
                push!(leaf_in, leaf)
            end
        end
        @assert length(leaf_in) == 2
        return leaf_in
    end
end

function find_leaf_of_soln(bbr::BlackBoxRegressor)
    leaf_in = 0
    if length(bbr.active_trees) == 1
        for (leaf, var) in bbr.leaf_variables
            if isapprox(getvalue(var), 1; atol=1e-5)
                leaf_in = leaf
                break
            end
        end
        leaf_in != 0 || throw(OCTException("BBR $(bbr.name) did not return a valid leaf. "*
                                           "Please make sure the MI constraints are added to a GM."))
        return leaf_in
    elseif length(bbr.active_trees) == 2
        leaf_in = []
        for (leaf, var) in bbr.leaf_variables
            if isapprox(getvalue(var), 1; atol=1e-5)
                push!(leaf_in, leaf)
            end
        end
        length(leaf_in) == 2 || throw(OCTException("BBR $(bbr.name) did not return valid leaves. "*
                                        "Please make sure the MI constraints are added to a GM."))
        return leaf_in
    end
end

""" 
    all_mi_constraints(bbl::BlackBoxLearner)

Returns all JuMP.ConstraintRefs associated with BBL. 
"""
function all_mi_constraints(bbl::BlackBoxLearner, hypertype::Union{String, Nothing} = nothing)
    all_constraints = []
    if hypertype isa Nothing
        for (leaf, constraints) in bbl.mi_constraints
            push!(all_constraints, constraints...)
        end
    elseif hypertype == "lower"
        for (leaf, constraints) in bbl.mi_constraints
            if leaf >= 0 
                push!(all_constraints, constraints...)
            end
        end
    elseif hypertype == "upper"
        for (leaf, constraints) in bbl.mi_constraints
            if leaf <= 0 
                push!(all_constraints, constraints...)
            end
        end
    end
    return all_constraints
end
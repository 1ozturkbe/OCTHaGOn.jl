
"""
Used to represent an abstract learner
"""
abstract type AbstractModel end



""" Contains data for a constraint that is repeated. """
@with_kw mutable struct LinkedClassifier
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    relax_var::Union{Real, JuMP.VariableRef} = 0.      # slack variable        
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}()
    leaf_variables::Dict = Dict{Int64, Tuple{JuMP.VariableRef, Array}}() 
    active_leaves::Array = []                          # leaf of last solution
    feas_gap::Array = []                               # Feasibility gaps of solutions   
    equality::Bool = false
end


""" Contains data for a constraint that is repeated. """
@with_kw mutable struct LinkedRegressor
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    dependent_var::JuMP.VariableRef                    # Dependent variable
    relax_var::Union{Real, JuMP.VariableRef} = 0.      # slack variable        
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}() # and their corresponding MI constraints,
    leaf_variables::Dict = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}() # and leaf variables. 
    active_leaves::Array = []                          # leaf of last solution    
    optima::Array = []
    actuals::Array = []
    feas_gap::Array = []                               # Feasibility gaps of solutions   
    equality::Bool = false
end


"""
    @with_kw mutable struct BlackBoxClassifier

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbls using functions:
    add_nonlinear_constraints
    add_nonlinear_or_compatible

Mandatory arguments are:
    vars::Array{JuMP.VariableRef,1}

Other arguments may be necessary for proper functioning:
    For data-driven constraints, need:
        X::DataFrame
        Y:: Array
    For constraint functions, need :
        constraint::Union{JuMP.ConstraintRef, Expr}

Optional arguments:
    expr_vars::Union{Array, Nothing}
        JuMP variables as function arguments (i.e. vars rolled up into vector forms).
        vars ⋐ flat(expr_vars)
    name::String
    equality::Bool
        Specifies whether function should be satisfied to an equality
"""
@with_kw mutable struct BlackBoxClassifier
    constraint::Union{Nothing, JuMP.ConstraintRef, Expr} # The "raw" constraint
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    name::String = ""                                  # Function name
    expr_vars::Array                                   # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    datamap::Union{Nothing,Array} = get_datamap(expr_vars, vars)     # ... with the required datamapping.
    f::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    g::Union{Nothing, Function} = gradientify(constraint, expr_vars)   # ... and its gradient f'n
    X::DataFrame = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)])
                                                       # Function samples
    Y::Array = []                                      # Function values
    gradients::Union{Nothing, DataFrame} = nothing     # Gradients
    curvatures::Union{Nothing, Array} = nothing        # Curvature around the points
    feas_ratio::Float64 = 0.                           # Feasible sample proportion
    convex::Bool = false
    local_convexity::Float64 = 0.
    vexity::Dict = Dict{Int64, Tuple}()                # Size and convexity of leaves
    equality::Bool = false                             # Equality check
    learners::Array{Union{IAI.OptimalTreeClassifier,
                          IAI.Heuristics.RandomForestClassifier,
                          AbstractModel}} = []    # Learners...
    learner_kwargs = []                                # And their kwargs... 
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}() # and their corresponding MI constraints,
    leaf_variables::Dict = Dict{Int64, Tuple{JuMP.VariableRef, Array}}() # and their leaf variables 
    active_leaves::Array = []                          # Leaf of last solution
    feas_gap::Array = []                               # Feasibility gaps of solutions   
    lls::Array{LinkedClassifier} = []                  # LinkedClassifiers
    relax_var::Union{Real, JuMP.VariableRef} = 0.    # slack variable        
    accuracies::Array{Float64} = []                    # and the tree misclassification scores.
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbc_defaults(length(vars))          # Relevant settings
    max_Y::Union{Nothing, Real} = nothing
    min_Y::Union{Nothing, Real} = nothing
    alg_list::Array{String} = ["OCT"]                  # List of algs used to approximate the constraint (e.g. 'CART','OCT','RF')
end


""" Superclass of LinkedClassifier and LinkedRegressor."""
LinkedLearner = Union{LinkedClassifier, LinkedRegressor}

"""
    @with_kw mutable struct BlackBoxRegressor

Allows for approximation of constraints using OCTs.
To be added to GlobalModel.bbls using functions:
    add_nonlinear_constraints
    add_nonlinear_or_compatible

    Mandatory arguments are:
        vars::Array{JuMP.VariableRef,1}
        dependent_var::JuMP.VariableRef

Other arguments may be necessary for proper functioning:
    For data-driven constraints, need:
        X::DataFrame
        Y:: Array
    For constraint functions, need :
        constraint::Union{JuMP.ConstraintRef, Expr}

Optional arguments:
    expr_vars::Union{Array, Nothing}
        JuMP variables as function arguments (i.e. vars rolled up into vector forms).
        vars ⋐ flat(expr_vars)
    name::String
    equality::Bool
        Specifies whether function should be satisfied to an equality
"""
@with_kw mutable struct BlackBoxRegressor
    constraint::Union{Nothing, JuMP.ConstraintRef, Expr}        # The "raw" constraint
    vars::Array{JuMP.VariableRef,1}                    # JuMP variables (flat)
    dependent_var::JuMP.VariableRef                    # Dependent variable
    name::String = ""                                  # Function name
    expr_vars::Array                                   # Function inputs (nonflat JuMP variables)
    varmap::Union{Nothing,Array} = get_varmap(expr_vars, vars)     # ... with the required varmapping.
    datamap::Union{Nothing,Array} = get_datamap(expr_vars, vars)     # ... with the required datamapping.
    f::Union{Nothing, Function} = functionify(constraint)         # ... and actually evaluated f'n
    g::Union{Nothing, Function} = gradientify(constraint, expr_vars)   # ... and its gradient f'n
    X::DataFrame = DataFrame(string.(vars) .=> [Float64[] 
    for i=1:length(vars)]) # Function samples
    Y::Array = []                                      # Function values
    gradients::Union{Nothing, DataFrame} = nothing     # Gradients 
    curvatures::Union{Nothing, Array} = nothing        # Curvature around the points
    infeas_X::DataFrame = DataFrame(string.(vars) .=> [Float64[] 
    for i=1:length(vars)]) # Infeasible samples, if any
    equality::Bool = false                             # Equality check
    learners::Array{Union{IAI.OptimalTreeRegressor, IAI.OptimalTreeClassifier,
                          IAI.Heuristics.RandomForestRegressor}} = []     # Learners...
    learner_kwargs = []                                # and their kwargs... 
    thresholds::Array{Pair} = []                       # For thresholding. 
    ul_data::Array{Dict} = Dict[]                      # Upper/lower bounding data
    active_trees::Dict{Int64, Union{Nothing, Pair}} = Dict() # Currently active tree indices
    mi_constraints::Dict = Dict{Int64, Array{JuMP.ConstraintRef}}() # and their corresponding MI constraints,
    leaf_variables::Dict = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}() # and leaf variables. 
    active_leaves::Array = []                          # leaf of last solution    
    optima::Array = []
    actuals::Array = []
    feas_gap::Array = []                               # Feasibility gaps of solutions   
    relax_var::Union{Real, JuMP.VariableRef} = 0.      # slack variable  
    accuracies::Array{Float64} = []                    # and the tree MSE scores.       
    lls::Array{LinkedRegressor} = []                   # Linked regressor mi_constraints and leaf_variables
    convex::Bool = false
    local_convexity::Float64 = 0.
    vexity::Dict = Dict{Int64, Tuple}()                # Size and convexity of leaves
    knn_tree::Union{KDTree, Nothing} = nothing         # KNN tree
    params::Dict = bbr_defaults(length(vars))          # Relevant settings
    max_Y::Union{Nothing, Real} = nothing
    min_Y::Union{Nothing, Real} = nothing
end

BlackBoxLearner = Union{BlackBoxClassifier, BlackBoxRegressor}

"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to use add_nonlinear_constraint to add bbls.
model must be a mixed integer convex model.
nonlinear_model can contain JuMP.NonlinearConstraints.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                            # Associated JuMP.Model
    name::String = "Model"                                       # Name
    bbls::Array{BlackBoxLearner} = BlackBoxLearner[]             # Constraints to be learned
    vars::Array{JuMP.VariableRef} = JuMP.all_variables(model)    # JuMP variables
    objective = JuMP.objective_function(model)                # Original objective function
    solution_history::DataFrame = DataFrame(string.(vars) .=> [Float64[] for i=1:length(vars)]) # Solution history
    cost::Array = []                                             # List of costs. 
    soldict::Dict = Dict()                                       # For solution extraction
    params::Dict = gm_defaults()                                 # GM settings
    og_objective = nothing                                       # Used to hold the original objective (LHS of dependent constraint)
end

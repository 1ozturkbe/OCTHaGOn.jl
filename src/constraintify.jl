""" 
    all_mi_constraints(bbl::BlackBoxLearner)

Returns all JuMP.ConstraintRefs associated with BBL. 
"""
function all_mi_constraints(bbl::BlackBoxLearner)
    all_constraints = []
    for (leaf, constraints) in bbl.mi_constraints
        push!(all_constraints, constraints...)
    end
    return all_constraints
end

""" 
    clear_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner)
    clear_tree_constraints!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    clear_tree_constraints!(gm::GlobalModel)

Clears the constraints bbl.mi_constraints 
as well as bbl.leaf_variables in GlobalModel. 
"""
function clear_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner)
    for constraint in all_mi_constraints(bbl)
        if is_valid(gm.model, constraint)
            delete(gm.model, constraint)
        end
    end
    bbl.mi_constraints = Dict{Int64, Array{JuMP.ConstraintRef}}()
    for (leaf, variable) in bbl.leaf_variables
        if is_valid(gm.model, variable)
            delete(gm.model, variable)
        end
    end
    bbl.leaf_variables = Dict{Int64, JuMP.VariableRef}()
    return
end

function clear_tree_constraints!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    for bbl in bbls
        clear_tree_constraints!(gm, bbl)
    end
    return
end

clear_tree_constraints!(gm::GlobalModel) = clear_tree_constraints!(gm, gm.bbls)

"""
    add_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner, M=1e5)
    add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner}; M=1e5)
    add_tree_constraints!(gm::GlobalModel)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier; M = 1e5)
    if bbc.feas_ratio == 1.0
        return
    elseif get_param(bbc, :reloaded)
        mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[end];
                                            M=M, equality = bbc.equality)
        merge!(bbc.mi_constraints, mi_constraints)
        merge!(bbc.leaf_variables, leaf_variables) # TODO: figure out issues to do with merging...
    elseif size(bbc.X, 1) == 0
        throw(OCTException("Constraint " * string(bbc.name) * " has not been sampled yet, and is thus untrained."))
    elseif isempty(bbc.learners)
        throw(OCTException("Constraint " * string(bbc.name) * " must be learned before tree constraints
                            can be generated."))
    elseif bbc.feas_ratio == 0.0
        throw(OCTException("Constraint " * string(bbc.name) * " is INFEASIBLE but you tried to include it in
            your global problem. Find at least one feasible solution, train and try again."))
    elseif !get_param(gm, :ignore_accuracy) && !check_accuracy(bbc)
        throw(OCTException("Constraint " * string(bbc.name) * " is inaccurately approximated. "))
    else
        mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[end];
                                            M=M, equality = bbc.equality);
        merge!(bbc.mi_constraints, mi_constraints)
        merge!(bbc.leaf_variables, leaf_variables) # TODO: figure out issues to do with merging...
    end
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor; M = 1e5)
    mi_constraints = Dict{Int64, Array{JuMP.ConstraintRef}}()
    leaf_variables = Dict{Int64, JuMP.VariableRef}()
    if get_param(bbr, :reloaded)
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                                   bbr.learners[end];
                                                                   M = M, equality = bbr.equality)
    elseif size(bbr.X, 1) == 0
        throw(OCTException("Constraint " * string(bbr.name) * " has not been sampled yet, and is thus untrained."))
    elseif isempty(bbr.learners)
        throw(OCTException("Constraint " * string(bbr.name) * " must be learned before tree constraints
                            can be generated."))
    elseif !isempty(bbr.ul_data[end])
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                                   bbr.learners[end], bbr.ul_data[end];
                                                                   M = M, equality = bbr.equality)
    elseif bbr.learners[end] isa IAI.OptimalTreeRegressor
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                                bbr.learners[end];
                                                                M = M, equality = bbr.equality)
    else
        throw(OCTException("Constraint " * string(bbr.name) * " is a Regressor, 
                            but doesn't have a ORT and/or OCT with upper/lower bounding approximators!"))
    end
    merge!(bbr.mi_constraints, mi_constraints)
    merge!(bbr.leaf_variables, leaf_variables)
    return
end

function add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner}; M = 1e5)
    for bbl in bbls
        add_tree_constraints!(gm, bbl; M = M)
    end
    return
end

add_tree_constraints!(gm::GlobalModel; M = 1e5) = add_tree_constraints!(gm, gm.bbls; M = M)

"""
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        lnr: A fitted IAI.OptimalTreeLearner
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
"""
function add_feas_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, lnr::IAI.OptimalTreeLearner;
                               M::Float64 = 1.e5, equality::Bool = false)
    check_if_trained(lnr);
    all_leaves = find_leaves(lnr)
    # Add a binary variable for each leaf
    feas_leaves = []
    if lnr isa IAI.OptimalTreeClassifier
        feas_leaves = 
        [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))];
    else 
        feas_leaves = all_leaves
    end
    constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in feas_leaves)
    constraints[0] = JuMP.ConstraintRef[]
    z_feas = @variable(m, [1:size(feas_leaves, 1)], Bin)
    leaf_variables = Dict{Int64, JuMP.VariableRef}(feas_leaves .=> z_feas)
    push!(constraints[0], @constraint(m, sum(z_feas) == 1))
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
    for i = 1:length(feas_leaves)
        leaf = feas_leaves[i]
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 - z_feas[i])))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold + M * (1 - z_feas[i]) >= sum(α .* x)))
        end
    end
    if equality
        infeas_leaves = [i for i in all_leaves if !Bool(IAI.get_classification_label(lnr, i))];
        z_infeas = @variable(m, [1:length(infeas_leaves)], Bin)
        push!(constraints[0], @constraint(m, sum(z_infeas) == 1))
        for leaf in infeas_leaves
            constraints[leaf] = JuMP.ConstraintRef[]
        end
        for i = 1:size(infeas_leaves, 1)
            leaf = infeas_leaves[i]
            leaf_variables[leaf] = z_infeas[i]
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 - z_infeas[i])))
            end
            for region in lowerDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold + M * (1 - z_infeas[i]) >= sum(α .* x)))
            end
        end
    end
    return constraints, leaf_variables
end

"""
    add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeRegressor;
                               M::Float64 = 1.e5, equality::Bool = false)

Creates a set of MIO constraints from a OptimalTreeRegressor
Arguments:
    m:: JuMP Model
    x:: independent JuMPVariable (features in lnr)
    y:: dependent JuMPVariable (output of lnr)
    lnr:: A fitted OptimalTreeRegressor
    M:: coefficient in bigM formulation
"""
function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeRegressor;
                               M::Float64 = 1.e5, equality::Bool = false)
    check_if_trained(lnr)
    all_leaves = find_leaves(lnr)
    # Add a binary variable for each leaf
    constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in all_leaves)
    constraints[0] = JuMP.ConstraintRef[]
    z = @variable(m, [1:size(all_leaves, 1)], Bin)
    leaf_variables = Dict{Int64, JuMP.VariableRef}(all_leaves .=> z)
    push!(constraints[0], @constraint(m, sum(z) == 1))
    # Getting lnr data
    pwlDict = pwl_constraint_data(lnr, Symbol.(x))
    upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
    for i = 1:size(all_leaves, 1)
        # ADDING CONSTRAINTS
        leaf = all_leaves[i]
        β0, β = pwlDict[leaf]
        push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M * (1 .- z[i])))
        if equality
            push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 + M * (1 .- z[i]) >= y))
        end
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 - z[i])))
        end
        for region in lowerDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold + M * (1 - z[i]) >= sum(α .* x)))
        end
    end
    return constraints, leaf_variables
end

"""
    add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeClassifier, 
            ul_data::Dict;
            M::Float64 = 1.e5, equality::Bool = false)

Creates a set of MIO constraints from a OptimalTreeClassifier that thresholds a BlackBoxRegressor.
Arguments:
    m:: JuMP Model
    x:: independent JuMPVariable (features in lnr)
    y:: dependent JuMPVariable (output of lnr)
    lnr:: A fitted OptimalTreeRegressor
    ul_data:: Upper and lower bounding hyperplanes for data in leaves of lnr
    M:: coefficient in bigM formulation
"""
function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeLearner, 
                               ul_data::Dict; M::Float64 = 1.e5, equality::Bool = false)
    constraints, leaf_variables = add_feas_constraints!(m, x, lnr, M=M, equality=equality)
    feas_leaves = collect(keys(leaf_variables))
    for leaf in feas_leaves
        (α0, α), (β0, β), (γ0, γ) = ul_data[leaf]
        push!(constraints[leaf], @constraint(m, y <= α0 + sum(α .* x) + M * (1 .- leaf_variables[leaf])))
        push!(constraints[leaf], @constraint(m, y + M * (1 .- leaf_variables[leaf]) >= β0 + sum(β .* x)))
        push!(constraints[leaf], @constraint(m, y + M * (1 .- leaf_variables[leaf]) >= γ0 + sum(γ .* x)))
        if equality
            push!(constraints[leaf], @constraint(m, y <= γ0 + sum(γ .* x) + M * (1 .- leaf_variables[leaf])))
        end    
    end
    return constraints, leaf_variables
end
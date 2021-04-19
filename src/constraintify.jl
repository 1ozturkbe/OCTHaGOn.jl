"""
    add_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner, M=1e5)
    add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner}; M=1e5)
    add_tree_constraints!(gm::GlobalModel)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners); M = 1e5)
    isempty(bbc.mi_constraints) || throw(OCTException("BBC $(bbc.name) already has associated MI approximation."))
    isempty(bbc.leaf_variables) || throw(OCTException("BBC $(bbc.name) already has associated MI variables."))
    if bbc.feas_ratio == 1.0
        z_feas = @variable(gm.model, binary = true)
        bbc.mi_constraints = Dict(1 => [@constraint(gm.model, z_feas == 1)])
        bbc.leaf_variables = Dict(1 => z_feas)
    elseif get_param(bbc, :reloaded)
        mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx];
                                            M=M, equality = bbc.equality)
        bbc.mi_constraints = mi_constraints
        bbc.leaf_variables = leaf_variables
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
        mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx];
                                            M=M, equality = bbc.equality);
        bbc.mi_constraints = mi_constraints
        bbc.leaf_variables = leaf_variables
    end
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners); M = 1e5)
    mi_constraints = Dict{Int64, Array{JuMP.ConstraintRef}}()
    leaf_variables = Dict{Int64, JuMP.VariableRef}()
    if size(bbr.X, 1) == 0 && !get_param(bbr, :reloaded)
        throw(OCTException("Constraint " * string(bbr.name) * " has not been sampled yet, and is thus untrained."))
    elseif bbr.convex
        mi_constraints = Dict(1 => [])
        for i = Int64.(floor.(size(bbr.X,1) .* rand(10)))
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
        end
        merge!(bbr.mi_constraints, mi_constraints)
        return
    elseif isempty(bbr.learners)
        throw(OCTException("Constraint " * string(bbr.name) * " must be learned before tree constraints
                            can be generated."))
    elseif isempty(bbr.ul_data)
        throw(OCTException("Constraint " * string(bbr.name) * " is a Regressor, 
        but doesn't have a ORT and/or OCT with upper/lower bounding approximators!"))
    end
    if bbr.thresholds[idx].first == "reg"
        (isempty(bbr.leaf_variables) || !isnothing(bbr.thresholds[idx].second)) ||
            throw(OCTException("Please clear previous tree constraints from $(gm.name) " *
                "before adding an unthresholded regression constraints."))
        if !isnothing(bbr.thresholds[idx].second) && !isempty(bbr.active_trees)
            bbr.thresholds[active_upper_tree(bbr)].second == bbr.thresholds[idx].second || 
                throw(OCTException("Upper-thresholded regressors must be preceeded by upper classifiers " * 
                                    "of the same threshold for $(bbr.name)."))
        end
    elseif bbr.thresholds[idx].first == "rfreg"
        isempty(bbr.leaf_variables) ||
            throw(OCTException("Please clear previous tree constraints from $(gm.name) " *
            "before adding an random forest regressor constraints."))
        isnothing(bbr.thresholds[idx].second) ||
            throw(OCTException("RandomForestRegressors are not allowed upper bounds."))
    elseif bbr.thresholds[idx].first == "upper"
        all(collect(keys(bbr.mi_constraints)) .>= 0)  || throw(OCTException("Please clear previous upper tree constraints from $(gm.name) " *
                                                          "before adding new constraints."))
    elseif bbr.thresholds[idx].first == "lower"
        all(collect(keys(bbr.mi_constraints)) .<= 0) || throw(OCTException("Please clear previous lower tree constraints from $(gm.name) " *
                                                            "before adding new constraints."))
    end
    if bbr.thresholds[idx].first != "rfreg"
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                                    bbr.learners[idx], bbr.ul_data[idx];
                                                                    M = M, equality = bbr.equality)
        if bbr.thresholds[idx].first == "upper"
            push!(mi_constraints[-1], @constraint(gm.model, bbr.dependent_var <= bbr.thresholds[idx].second))
        elseif bbr.thresholds[idx].first == "lower"
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= bbr.thresholds[idx].second))
        end
        merge!(bbr.mi_constraints, mi_constraints)
        merge!(bbr.leaf_variables, leaf_variables)
        bbr.active_trees[idx] = bbr.thresholds[idx]
    else
        trees = get_random_trees(bbr.learners[idx])
        for i=1:length(trees)
            mic, miv = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                trees[i], bbr.ul_data[idx][i];
                                                M = M, equality = bbr.equality)                                     
            merge!(mi_constraints, mic)
            merge!(leaf_variables, miv)
        end
        merge!(bbr.mi_constraints, mi_constraints)
        merge!(bbr.leaf_variables, leaf_variables)
        bbr.active_trees[idx] = bbr.thresholds[idx]    
    end
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
    constraints[1] = JuMP.ConstraintRef[]
    z_feas = @variable(m, [1:size(feas_leaves, 1)], Bin)
    leaf_variables = Dict{Int64, JuMP.VariableRef}(feas_leaves .=> z_feas)
    push!(constraints[1], @constraint(m, sum(z_feas) == 1))
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
        push!(constraints[1], @constraint(m, sum(z_infeas) == 1))
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
function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, 
                               lnr::IAI.OptimalTreeLearner,
                               ul_data::Dict; M::Float64 = 1.e5, equality::Bool = false)
    if lnr isa OptimalTreeRegressor                
        check_if_trained(lnr)
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in all_leaves)
        constraints[1] = JuMP.ConstraintRef[]
        z = @variable(m, [1:size(all_leaves, 1)], Bin)
        leaf_variables = Dict{Int64, JuMP.VariableRef}(all_leaves .=> z)
        push!(constraints[1], @constraint(m, sum(z) == 1))
        # Getting lnr data
        pwlDict = pwl_constraint_data(lnr, Symbol.(x))
        upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
        for i = 1:size(all_leaves, 1)
            # ADDING CONSTRAINTS
            leaf = all_leaves[i]
            β0, β = pwlDict[leaf]
            if !isempty(ul_data) # Overwrite if we have a lower approximator
                α0, α = ul_data[-leaf]
                constraints[-leaf] = [@constraint(m, sum(α .* x) + α0 + M * (1 .- z[i]) >= y)] #TODO: could be problematic
                β0, β = ul_data[leaf]
            end
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
    elseif lnr isa OptimalTreeClassifier
        constraints, leaf_variables = add_feas_constraints!(m, x, lnr, M=M, equality=equality)
        if !isempty(ul_data)
            if all(keys(ul_data) .<= 0) || !all(keys(ul_data) .>= 0) # means an upper or upperlower bounding tree
                constraints = Dict(-key => value for (key, value) in constraints) # hacky sign flipping for upkeep. 
                leaf_variables = Dict(-key => value for (key, value) in leaf_variables) 
            end
            for leaf in collect(keys(ul_data))
                γ0, γ = ul_data[leaf]
                if !haskey(constraints, leaf) # occurs with ORTS with bounding hyperplanes
                    constraints[leaf] = [@constraint(m, y <= γ0 + sum(γ .* x) + M * (1 .- leaf_variables[-leaf]))]
                elseif leaf <= 0
                    push!(constraints[leaf], @constraint(m, y <= γ0 + sum(γ .* x) + M * (1 .- leaf_variables[leaf])))
                else
                    push!(constraints[leaf], @constraint(m, y + M * (1 .- leaf_variables[leaf]) >= γ0 + sum(γ .* x)))
                end
            end
        end
        return constraints, leaf_variables
    end
end

function clear_upper_constraints!(gm, bbr::BlackBoxRegressor)
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key <= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTException("Constraints in BlackBoxRegressor $(bbr.name) " * 
                                    " could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, leaf_var) in bbr.leaf_variables
        if leaf_key <= 0
            if is_valid(gm.model, leaf_var)
                delete(gm.model, leaf_var)
                delete!(bbr.leaf_variables, leaf_key)
            else
                throw(OCTException("Variables in BlackBoxRegressor $(bbr.name) " * 
                " could not be removed."))
            end
        end
    end
    idx = active_upper_tree(bbr)
    delete!(bbr.active_trees, idx)
    return
end

function clear_lower_constraints!(gm, bbr::BlackBoxRegressor)
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key >= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTException("Constraints in BlackBoxRegressor $(bbr.name) " * 
                                    " could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, leaf_var) in bbr.leaf_variables
        if leaf_key >= 0
            if is_valid(gm.model, leaf_var)
                delete(gm.model, leaf_var)
                delete!(bbr.leaf_variables, leaf_key)
            else
                throw(OCTException("Variables in BlackBoxRegressor $(bbr.name) " * 
                " could not be removed."))
            end
        end
    end
    idx = active_lower_tree(bbr)
    delete!(bbr.active_trees, idx)
    return
end

""" 
    clear_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier)
    clear_tree_constraints!(gm::GlobalModel, bbc::BlackBoxRegressor)
    clear_tree_constraints!(gm::GlobalModel, bbls::Array{BlackBoxLearner})
    clear_tree_constraints!(gm::GlobalModel)

Clears the constraints bbl.mi_constraints 
as well as bbl.leaf_variables in GlobalModel. 
"""
function clear_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier)
    for (leaf_key, leaf_constrs) in bbc.mi_constraints
        while !isempty(leaf_constrs)
            constr = pop!(leaf_constrs)
            if is_valid(gm.model, constr)
                delete(gm.model, constr)
            else
                push!(leaf_constrs, constr) # make sure to put the constraint back. 
                throw(OCTException("Constraints in BlackBoxClassifier $(bbr.name) " * 
                                " could not be removed."))
            end
        end
        delete!(bbc.mi_constraints, leaf_key)
    end
    for (leaf_key, leaf_var) in bbc.leaf_variables
        if is_valid(gm.model, leaf_var)
            delete(gm.model, leaf_var)
            delete!(bbc.leaf_variables, leaf_key)
        else
            throw(OCTException("Variables in BlackBoxClassifier $(bbr.name) " * 
            " could not be removed."))
        end
    end
    return
end

function clear_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor)
    clear_lower_constraints!(gm, bbr)
    clear_upper_constraints!(gm, bbr)
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
    update_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    update_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))

Updates the MI constraints associated with a BBL. 
For BBRs, makes sure to replace the appropriate lower/upper/regressor approximations. 
"""
function update_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    if isempty(bbr.active_trees) # no mi constraints yet. 
        add_tree_constraints!(gm, bbr, idx)
    elseif length(bbr.active_trees) == 1 # one set of mi_constraints
        new_threshold = bbr.thresholds[idx]
        active_idx = collect(keys(bbr.active_trees))[1]
        last_threshold = bbr.active_trees[active_idx]
        if new_threshold.first == last_threshold.first
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        elseif new_threshold == Pair("reg", nothing) || last_threshold == Pair("reg", nothing) || 
            last_threshold.first == "upperlower" || new_threshold.first == "upperlower"
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        elseif new_threshold.first == "reg" && last_threshold.first == "upper" || 
                last_threshold.first == "reg" && new_threshold.first == "upper"
            if new_threshold.second == last_threshold.second # if corresponding upper thresholds. 
                add_tree_constraints!(gm, bbr, idx)
                return
            else
                clear_tree_constraints!(gm, bbr)
                add_tree_constraints!(gm, bbr, idx)
                return
            end
        elseif last_threshold.first in valid_uppers && new_threshold.first in valid_lowers || 
            last_threshold.first in valid_lowers && new_threshold.first in valid_uppers
            add_tree_constraints!(gm, bbr, idx)
            return
        else # updating a lower or upper threshold
            clear_tree_constraints!(gm, bbr)
            add_tree_constraints!(gm, bbr, idx)
            return
        end
    elseif length(bbr.active_trees) == 2 # two sets of mi_constraints
        if bbr.thresholds[idx] == Pair("reg", nothing) || bbr.thresholds[idx].first == "upperlower"
            clear_tree_constraints!(gm, bbr)                                                  
            add_tree_constraints!(gm, bbr, idx)                                               
            return
        end
        hypertype = bbr.thresholds[idx].first # otherwise, check approximation type
        if hypertype in valid_lowers
            clear_lower_constraints!(gm, bbr)
        elseif hypertype in valid_uppers
            clear_upper_constraints!(gm, bbr)
        else
            throw(OCTException("Hypertype $(hypertype) not recognized."))
        end
        add_tree_constraints!(gm, bbr, idx)
        return 
    else
        throw(OCTException("$(gm.name) has too many active lower/upper bounding regressors."))
    end
end

function update_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    clear_tree_constraints!(gm, bbc)
    add_tree_constraints!(gm, bbc, idx)
    return
end
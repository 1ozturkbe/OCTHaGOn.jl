"""
    add_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner)
    add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner})
    add_tree_constraints!(gm::GlobalModel)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    isempty(bbc.mi_constraints) || throw(OCTException("BBC $(bbc.name) already has associated MI approximation."))
    isempty(bbc.leaf_variables) || throw(OCTException("BBC $(bbc.name) already has associated MI variables."))
    if bbc.feas_ratio == 1.0
        z_feas = @variable(gm.model, binary = true)
        bbc.mi_constraints = Dict(1 => [@constraint(gm.model, z_feas == 1)])
        bbc.leaf_variables = Dict(1 => z_feas)
    elseif get_param(bbc, :reloaded)
        mi_constraints, leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx];
                                            M = bbc.M, equality = bbc.equality, lcs = bbc.lls)
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
                                            M = bbc.M, equality = bbc.equality, lcs = bbc.lls);
        bbc.mi_constraints = mi_constraints
        bbc.leaf_variables = leaf_variables
    end
    bbc.active_leaves = []
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    mi_constraints = Dict{Int64, Array{JuMP.ConstraintRef}}()
    leaf_variables = Dict{Int64, JuMP.VariableRef}()
    if size(bbr.X, 1) == 0 && !get_param(bbr, :reloaded)
        throw(OCTException("Constraint " * string(bbr.name) * " has not been sampled yet, and is thus untrained."))
    elseif bbr.convex
        clear_tree_constraints!(gm, bbr)
        mi_constraints = Dict(1 => [])
        for i = Int64.(ceil.(size(bbr.X,1) .* rand(10)))
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
        end
        merge!(append!, bbr.mi_constraints, mi_constraints)
        if get_param(bbr, :linked)
            for lr in bbr.lls
                mc = Dict(1 => [])
                for i = Int64.(ceil.(size(bbr.X,1) .* rand(10)))
                    push!(mc[1], @constraint(gm.model, lr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (lr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
                end
                merge!(append!, lr.mi_constraints, mc)
                lr.active_leaves = [1]
            end
        end
        bbr.active_leaves = [1]
        return
    elseif isempty(bbr.learners)
        throw(OCTException("Constraint " * string(bbr.name) * " must be learned before tree constraints
                            can be generated."))
    elseif isempty(bbr.ul_data)
        throw(OCTException("Constraint " * string(bbr.name) * " is a Regressor, 
        but doesn't have a ORT and/or OCT with upper/lower bounding approximators!"))
    end
    if get_param(bbr, :linked)
        bbr.thresholds[idx].first == "reg" || 
            throw(OCTException("BBRs with LinkedRegressors are only allowed ORT approximators."))
        length(bbr.active_trees) == 0 || 
            throw(OCTException("BBRs with LinkedRegressors cannot have more than one active tree."))
    elseif bbr.thresholds[idx].first == "reg"
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
        get_param(bbr, :linked) && throw(OCTException("Random Forests cannot be used to approximate " *
                        "linked regressor $(bbr.name)."))
        isnothing(bbr.thresholds[idx].second) ||
            throw(OCTException("RandomForestRegressors are not allowed upper bounds."))
    elseif bbr.thresholds[idx].first == "upper"
        all(collect(keys(bbr.mi_constraints)) .>= 0)  || throw(OCTException("Please clear previous upper tree constraints from $(gm.name) " *
                                                          "before adding new constraints."))
    elseif bbr.thresholds[idx].first == "lower"
        all(collect(keys(bbr.mi_constraints)) .<= 0) || throw(OCTException("Please clear previous lower tree constraints from $(gm.name) " *
                                                            "before adding new constraints."))
    end
    if bbr.thresholds[idx].first == "rfreg"
        # NOTE: RFREG augments LinkedLearners. THIS WILL CAUSE BUGS. NOT A FULLY SUPPORTED FEATURE. 
        trees = get_random_trees(bbr.learners[idx])
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                    trees[1], bbr.ul_data[idx][1]; M = bbr.M, equality = bbr.equality) 
        for i=2:length(trees)
            mic, miv = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                trees[i], bbr.ul_data[idx][i];
                                                M = bbr.M, equality = bbr.equality) 
            nll = LinkedRegressor(vars = bbr.vars, dependent_var = bbr.dependent_var)
            merge!(append!, nll.mi_constraints, mic)
            merge!(append!, nll.leaf_variables, miv)       
            push!(bbr.lls, nll)
        end
        merge!(append!, bbr.mi_constraints, mi_constraints)
        merge!(append!, bbr.leaf_variables, leaf_variables)
        bbr.active_trees[idx] = bbr.thresholds[idx]    
    else
        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                bbr.learners[idx], bbr.ul_data[idx];
                                                M = bbr.M, equality = bbr.equality, lrs = bbr.lls)
        if bbr.thresholds[idx].first == "upper"
            push!(mi_constraints[-1], @constraint(gm.model, bbr.dependent_var <= bbr.thresholds[idx].second))
        elseif bbr.thresholds[idx].first == "lower"
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= bbr.thresholds[idx].second))
        end
        merge!(append!, bbr.mi_constraints, mi_constraints)
        merge!(append!, bbr.leaf_variables, leaf_variables)
        bbr.active_trees[idx] = bbr.thresholds[idx]
    end
    bbr.active_leaves = []
    return
end

function add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner})
    for bbl in bbls
        add_tree_constraints!(gm, bbl)
    end
    return
end

add_tree_constraints!(gm::GlobalModel) = add_tree_constraints!(gm, gm.bbls)

"""
    Creates a set of binary feasibility constraints from
    a binary classification tree:
    Arguments:
        lnr: A fitted IAI.OptimalTreeLearner
        m:: JuMP Model
        x:: JuMPVariables (features in lnr)
"""
function add_feas_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, lnr::IAI.OptimalTreeLearner;
                               M::Real = 1.e5, equality::Bool = false, 
                               relax_var::Union{Real, JuMP.VariableRef} = 0,
                               lcs::Array = [])
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
    for lc in lcs
        lc_z_feas = @variable(m, [1:size(feas_leaves, 1)], Bin)
        lc.leaf_variables = Dict{Int64, JuMP.VariableRef}(feas_leaves .=> lc_z_feas)
        lc.mi_constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in feas_leaves)
        lc.mi_constraints[1] = [@constraint(m, sum(lc_z_feas) == 1)]
        lc.active_leaves = []
    end
    # Getting lnr data
    upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
    for i = 1:length(feas_leaves)
        leaf = feas_leaves[i]
        # ADDING TRUST REGIONS
        for region in upperDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 + relax_var - z_feas[i])))
            for lc in lcs
                push!(lc.mi_constraints[leaf], @constraint(m, threshold <= sum(α .* lc.vars) + 
                            M * (1 + lc.relax_var - lc.leaf_variables[leaf])))
            end
        end
        for region in lowerDict[leaf]
            threshold, α = region
            push!(constraints[leaf], @constraint(m, threshold + M * (1 + relax_var - z_feas[i]) >= sum(α .* x)))
            for lc in lcs
                push!(lc.mi_constraints[leaf], @constraint(m, threshold + M * (1 + lc.relax_var - lc.leaf_variables[leaf]) >= 
                                                                sum(α .* lc.vars)))
            end
        end
    end
    if equality
        infeas_leaves = [i for i in all_leaves if !Bool(IAI.get_classification_label(lnr, i))];
        z_infeas = @variable(m, [1:length(infeas_leaves)], Bin)
        push!(constraints[1], @constraint(m, sum(z_infeas) == 1))
        for leaf in infeas_leaves
            constraints[leaf] = JuMP.ConstraintRef[]
        end
        for lc in lcs
            for lc in lcs
                lc_z_infeas = @variable(m, [1:length(infeas_leaves)], Bin)
                for i=1:length(infeas_leaves)
                    lc.leaf_variables[infeas_leaves[i]] = lc_z_infeas[i]
                    lc.mi_constraints[infeas_leaves[i]] = JuMP.ConstraintRef[] 
                    push!(lc.mi_constraints[1], @constraint(m, sum(lc_z_infeas) == 1))
                end
            end
        end
        for i = 1:size(infeas_leaves, 1)
            leaf = infeas_leaves[i]
            leaf_variables[leaf] = z_infeas[i]
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 + relax_var - z_infeas[i])))
                for lc in lcs
                    push!(lc.mi_constraints[leaf], @constraint(m, threshold <= sum(α .* lc.vars) + 
                                M * (1 + lc.relax_var - lc.leaf_variables[leaf])))
                end
            end
            for region in lowerDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold + M * (1 + relax_var - z_infeas[i]) >= sum(α .* x)))
                for lc in lcs
                    push!(lc.mi_constraints[leaf], @constraint(m, threshold + M * (1 + lc.relax_var - lc.leaf_variables[leaf]) >= 
                                                                    sum(α .* lc.vars)))
                end
            end
        end
    end
    return constraints, leaf_variables
end

"""
    add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeClassifier, 
            ul_data::Dict;
            M::Real = 1.e5, equality::Bool = false)

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
                               ul_data::Dict; 
                               M::Real = 1.e5, equality::Bool = false, 
                               relax_var::Union{Real, JuMP.VariableRef} = 0,
                               lrs::Array = [])
    # TODO: Determine whether I should relax approximator or trust regions or both. 
    if lnr isa OptimalTreeRegressor                
        check_if_trained(lnr)
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in all_leaves)
        constraints[1] = JuMP.ConstraintRef[]
        z = @variable(m, [1:size(all_leaves, 1)], Bin)
        leaf_variables = Dict{Int64, JuMP.VariableRef}(all_leaves .=> z)
        push!(constraints[1], @constraint(m, sum(z) == 1))
        for lr in lrs
            @assert isempty(lr.mi_constraints) # TODO: remove check after testing
            @assert isempty(lr.leaf_variables) # TODO: remove check after testing
            lr_z = @variable(m, [1:size(all_leaves, 1)], Bin)
            lr.leaf_variables = Dict{Int64, JuMP.VariableRef}(all_leaves .=> lr_z)
            lr.mi_constraints = Dict(leaf => JuMP.ConstraintRef[] for leaf in all_leaves)
            lr.mi_constraints[1] = [@constraint(m, sum(lr_z) == 1)]
            lr.active_leaves = []
        end
        # Getting lnr data
        pwlDict = pwl_constraint_data(lnr, Symbol.(x))
        upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
        for i = 1:size(all_leaves, 1)
            # ADDING CONSTRAINTS
            leaf = all_leaves[i]
            β0, β = pwlDict[leaf]
            if equality
                # Use the ridge regressor!
                push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 + M * (1 + relax_var .- z[i]) >= y))
                [push!(lr.mi_constraints[leaf],  @constraint(m, sum(β .* lr.vars) + 
                    β0 + M * (1 + lr.relax_var .- lr.leaf_variables[leaf]) >= lr.dependent_var)) for lr in lrs]
                push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M * (1 + relax_var .- z[i])))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.vars) + β0 <= 
                        lr.dependent_var + M * (1 + lr.relax_var .- lr.leaf_variables[leaf]))) for lr in lrs]
                # And use the lower bound
                β0, β = ul_data[leaf]
                push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M * (1 + relax_var .- z[i])))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.vars) + β0 <= 
                        lr.dependent_var + M * (1 + lr.relax_var.- lr.leaf_variables[leaf]))) for lr in lrs]
            elseif !isempty(ul_data) 
                # Use only the lower approximator
                β0, β = ul_data[leaf] # update the lower approximator
                push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M * (1 + relax_var.- z[i])))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.vars) + β0 <= 
                        lr.dependent_var + M * (1 + lr.relax_var .- lr.leaf_variables[leaf]))) for lr in lrs]
            else 
                # Use only the ridge regressor as the lower approximator. 
                push!(constraints[leaf], @constraint(m, sum(β .* x) + β0 <= y + M * (1 + relax_var .- z[i])))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.vars) + β0 <= 
                        lr.dependent_var + M * (1 + lr.relax_var .- lr.leaf_variables[leaf]))) for lr in lrs]
            end
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold <= sum(α .* x) + M * (1 + relax_var - z[i])))
                [push!(lr.mi_constraints[leaf], @constraint(m, threshold <= sum(α .* lr.vars) + 
                    M * (1 + lr.relax_var - lr.leaf_variables[leaf]))) for lr in lrs]
            end
            for region in lowerDict[leaf]
                threshold, α = region
                push!(constraints[leaf], @constraint(m, threshold + M * (1 + relax_var - z[i]) >= sum(α .* x)))
                [push!(lr.mi_constraints[leaf], @constraint(m, threshold + M * (1 + lr.relax_var - lr.leaf_variables[leaf]) >= 
                    sum(α .* lr.vars))) for lr in lrs]
            end
        end
        return constraints, leaf_variables
    elseif lnr isa OptimalTreeClassifier
        isempty(lrs) || throw(OCTException("Bug: Cannot use OCTs to approximate linked BBRs."))
        constraints, leaf_variables = add_feas_constraints!(m, x, lnr, M = M, equality = equality, lcs = lrs)
        if !isempty(ul_data)
            if all(keys(ul_data) .<= 0) || !all(keys(ul_data) .>= 0) # means an upper or upperlower bounding tree
                constraints = Dict(-key => value for (key, value) in constraints) # hacky sign flipping for upkeep. 
                leaf_variables = Dict(-key => value for (key, value) in leaf_variables) 
            end
            for leaf in collect(keys(ul_data))
                γ0, γ = ul_data[leaf]
                if !haskey(constraints, leaf) # occurs with ORTS with bounding hyperplanes
                    constraints[leaf] = [@constraint(m, y <= γ0 + sum(γ .* x) + M * (1 + relax_var .- leaf_variables[-leaf]))]
                elseif leaf <= 0
                    push!(constraints[leaf], @constraint(m, y <= γ0 + sum(γ .* x) + M * (1 + relax_var .- leaf_variables[leaf])))
                else
                    push!(constraints[leaf], @constraint(m, y + M * (1 + relax_var .- leaf_variables[leaf]) >= γ0 + sum(γ .* x)))
                end
            end
        end
        return constraints, leaf_variables
    end
end

""" Clears upper-bounding constraints from a BBR and its associated GM. """
function clear_upper_constraints!(gm, bbr::Union{BlackBoxRegressor, LinkedRegressor})
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key <= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTException("Bug: Constraints could not be removed."))
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
                throw(OCTException("Bug: Variables could not be removed."))
            end
        end
    end
    if bbr isa BlackBoxRegressor
        idx = active_upper_tree(bbr)
        delete!(bbr.active_trees, idx)
        for lr in bbr.lls
            clear_upper_constraints!(gm, lr)
        end
    end
    return
end

""" Clears lower-bounding constraints from a BBR and its associated GM. """
function clear_lower_constraints!(gm, bbr::Union{BlackBoxRegressor, LinkedRegressor})
    for (leaf_key, leaf_constrs) in bbr.mi_constraints
        if leaf_key >= 0
            while !isempty(leaf_constrs)
                constr = pop!(leaf_constrs)
                if is_valid(gm.model, constr)
                    delete(gm.model, constr)
                else
                    push!(leaf_constrs, constr) # make sure to put the constraint back. 
                    throw(OCTException("Bug: Constraints could not be removed."))
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
                throw(OCTException("Bug: Variables could not be removed."))
            end
        end
    end
    if bbr isa BlackBoxRegressor
        idx = active_lower_tree(bbr)
        delete!(bbr.active_trees, idx)
        for lr in bbr.lls
            clear_lower_constraints!(gm, lr)
        end
    end
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
function clear_tree_constraints!(gm::GlobalModel, bbc::Union{BlackBoxClassifier, LinkedClassifier})
    for (leaf_key, leaf_constrs) in bbc.mi_constraints
        while !isempty(leaf_constrs)
            constr = pop!(leaf_constrs)
            if is_valid(gm.model, constr)
                delete(gm.model, constr)
            else
                push!(leaf_constrs, constr) # make sure to put the constraint back. 
                throw(OCTException("Bug: Constraints could not be removed."))
            end
        end
        delete!(bbc.mi_constraints, leaf_key)
    end
    for (leaf_key, leaf_var) in bbc.leaf_variables
        if is_valid(gm.model, leaf_var)
            delete(gm.model, leaf_var)
            delete!(bbc.leaf_variables, leaf_key)
        else
            throw(OCTException("Bug: Variables could not be removed."))
        end
    end
    bbc.active_leaves = []
    if bbc isa BlackBoxClassifier
        for lc in bbc.lls
            clear_tree_constraints!(gm, lc)
        end
    end
    return
end

function clear_tree_constraints!(gm::GlobalModel, bbr::Union{BlackBoxRegressor, LinkedRegressor})
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
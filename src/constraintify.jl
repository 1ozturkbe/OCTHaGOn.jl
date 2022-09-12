""" 
    bounded_aux(x::Array{JuMP.VariableRef}, binary_var::JuMP.VariableRef)
    bounded_aux(x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, binary_var::JuMP.VariableRef)
Generates binary-bounded auxiliary variables and their bounding constraints of the same size as x + y.
"""
function bounded_aux(x::Array{JuMP.VariableRef}, binary_var::JuMP.VariableRef)
    var_bounds = get_bounds(x)
    aux_vars = @variable(x[1].model, [1:length(x)])
    bound_cons = []
    for i = 1:length(x) # Bounding auxiliary variables
        bds = var_bounds[x[i]]
        push!(bound_cons, @constraint(x[i].model, aux_vars[i] >= minimum(bds)*binary_var))
        push!(bound_cons, @constraint(x[i].model, aux_vars[i] <= maximum(bds)*binary_var))
    end
    aux_vars = (binary_var, aux_vars)
    return aux_vars, bound_cons
end

function bounded_aux(x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, binary_var::JuMP.VariableRef)
    aux_vars, bound_cons = bounded_aux(x, binary_var)
    aux_dep = @variable(y.model)
    push!(bound_cons, @constraint(y.model, aux_dep >= binary_var * JuMP.lower_bound(y)))
    push!(bound_cons, @constraint(y.model, aux_dep <= binary_var * JuMP.upper_bound(y)))
    aux_vars = (aux_vars[1], aux_vars[2], aux_dep)
    return aux_vars, bound_cons
end

"""
    add_tree_constraints!(gm::GlobalModel, bbl::BlackBoxLearner)
    add_tree_constraints!(gm::GlobalModel, bbls::Vector{BlackBoxLearner})
    add_tree_constraints!(gm::GlobalModel)

Generates MI constraints from gm.learners, and adds them to gm.model.
"""
function add_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    isempty(bbc.mi_constraints) || throw(OCTHaGOnException("BBC $(bbc.name) already has associated MI approximation."))
    isempty(bbc.leaf_variables) || throw(OCTHaGOnException("BBC $(bbc.name) already has associated MI variables."))

    relax_var = isnothing(gm.relax_var) ? 0 : gm.relax_var;
    
    if bbc.feas_ratio == 1.0 # Just a placeholder to show that the tree is "trained". 
        z_feas = @variable(gm.model, binary = true)
        bbc.mi_constraints = Dict(1 => [@constraint(gm.model, z_feas == 1)])
        bbc.leaf_variables = Dict(1 => (z_feas, []))
    elseif get_param(bbc, :reloaded)
        lnr = bbc.learners[idx]
        if lnr isa AbstractClassifier
            bbc.mi_constraints, bbc.leaf_variables = embed_mio!(lnr, gm, bbc)
        else
            bbc.mi_constraints, bbc.leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx], bbc;
                                                equality = bbc.equality, relax_var = relax_var,
                                                lcs = bbc.lls, ro_factor = get_param(gm, :ro_factor))
        end
    elseif size(bbc.X, 1) == 0
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " has not been sampled yet, and is thus untrained."))
    elseif isempty(bbc.learners)
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " must be learned before tree constraints
                            can be generated."))
    elseif bbc.feas_ratio == 0.0
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " is INFEASIBLE but you tried to include it in
            your global problem. Find at least one feasible solution, train and try again."))
    elseif !get_param(gm, :ignore_accuracy) && !check_accuracy(bbc)
        throw(OCTHaGOnException("Constraint " * string(bbc.name) * " is inaccurately approximated. "))
    else
        lnr = bbc.learners[idx]
        if lnr isa AbstractClassifier
            bbc.mi_constraints, bbc.leaf_variables = embed_mio!(lnr, gm, bbc)
        else
            bbc.mi_constraints, bbc.leaf_variables = add_feas_constraints!(gm.model, bbc.vars, bbc.learners[idx], bbc;
                                                equality = bbc.equality, relax_var=relax_var, 
                                                lcs = bbc.lls, ro_factor = get_param(gm, :ro_factor))
        end
    end
    bbc.active_leaves = []
    return
end

function add_tree_constraints!(gm::GlobalModel, bbr::BlackBoxRegressor, idx = length(bbr.learners))
    mi_constraints = Dict{Int64, Array{JuMP.ConstraintRef}}()
    leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}()
    if size(bbr.X, 1) == 0 && !get_param(bbr, :reloaded)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " has not been sampled yet, and is thus untrained."))
    elseif bbr.convex && !bbr.equality
        clear_tree_constraints!(gm, bbr)
        mi_constraints = Dict(1 => [])
        for i = Int64.(ceil.(size(bbr.X,1) .* rand(10)))
            push!(mi_constraints[1], @constraint(gm.model, bbr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (bbr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
        end
        merge!(append!, bbr.mi_constraints, mi_constraints)
        if !isempty(bbr.lls)
            for lr in bbr.lls
                mc = Dict(1 => [])
                # Number of initial cuts depends on the dimension of the constraint
                pt_idxs = Int64.(ceil.(size(bbr.X,1) .* rand(maximum([10, Int64(10*ceil(log(length(bbr.vars))))]))))
                update_gradients(bbr, pt_idxs)
                for i in pt_idxs
                    push!(mc[1], @constraint(gm.model, lr.dependent_var >= sum(Array(bbr.gradients[i,:]) .* (lr.vars .- Array(bbr.X[i, :]))) + bbr.Y[i]))
                end
                merge!(append!, lr.mi_constraints, mc)
                lr.active_leaves = [1]
            end
        end
        bbr.active_leaves = [1]
        return
    elseif isempty(bbr.learners)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " must be learned before tree constraints
                            can be generated."))
    elseif isempty(bbr.ul_data) && !(bbr.learners[idx] isa AbstractRegressor)
        throw(OCTHaGOnException("Constraint " * string(bbr.name) * " is a Regressor, 
        but doesn't have a ORT and/or OCT with upper/lower bounding approximators!"))

    elseif bbr.learners[idx] isa AbstractRegressor
        mi_constraints, leaf_variables = embed_mio!(bbr.learners[idx], gm, bbr)
        merge!(append!, bbr.mi_constraints, mi_constraints)
        merge!(append!, bbr.leaf_variables, leaf_variables)

        bbr.active_leaves = []
        return
    end
    if !isempty(bbr.lls)
        bbr.thresholds[idx].first == "reg" || 
            throw(OCTHaGOnException("BBRs with LinkedRegressors are only allowed ORT approximators."))
        length(bbr.active_trees) == 0 || 
            throw(OCTHaGOnException("BBRs with LinkedRegressors cannot have more than one active tree."))
    elseif bbr.thresholds[idx].first == "reg"
        (isempty(bbr.leaf_variables) || !isnothing(bbr.thresholds[idx].second)) ||
            throw(OCTHaGOnException("Please clear previous tree constraints from $(gm.name) " *
                "before adding an unthresholded regression constraints."))
        if !isnothing(bbr.thresholds[idx].second) && !isempty(bbr.active_trees)
            bbr.thresholds[active_upper_tree(bbr)].second == bbr.thresholds[idx].second || 
                throw(OCTHaGOnException("Upper-thresholded regressors must be preceeded by upper classifiers " * 
                                    "of the same threshold for $(bbr.name)."))
        end
    elseif bbr.thresholds[idx].first == "rfreg"
        isempty(bbr.leaf_variables) ||
            throw(OCTHaGOnException("Please clear previous tree constraints from $(gm.name) " *
            "before adding an random forest regressor constraints."))
            !isempty(bbr.lls) && throw(OCTHaGOnException("Random Forests cannot be used to approximate " *
                        "linked BBR $(bbr.name)."))
        isnothing(bbr.thresholds[idx].second) ||
            throw(OCTHaGOnException("RandomForestRegressors are not allowed upper bounds."))
        bbr.equality && 
            throw(OCTHaGOnException("RandomForest cannot approximate BBR $(bbr.name) since it is an equality constraint."))
    elseif bbr.thresholds[idx].first == "upper"
        all(collect(keys(bbr.mi_constraints)) .>= 0)  || throw(OCTHaGOnException("Please clear previous upper tree constraints from $(gm.name) " *
                                                          "before adding new constraints."))
    elseif bbr.thresholds[idx].first == "lower"
        all(collect(keys(bbr.mi_constraints)) .<= 0) || throw(OCTHaGOnException("Please clear previous lower tree constraints from $(gm.name) " *
                                                            "before adding new constraints."))
    end
    if bbr.thresholds[idx].first == "rfreg"
        # NOTE: RFREG augments LinkedLearners. THIS WILL CAUSE BUGS. NOT A FULLY SUPPORTED FEATURE. 
        trees = get_random_trees(bbr.learners[idx])
        bbr.mi_constraints, bbr.leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                    trees[1], bbr, bbr.ul_data[idx][1]; equality = bbr.equality) 
        for i=2:length(trees)
            mic, lv = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, 
                                                trees[i], bbr, bbr.ul_data[idx][i];
                                                equality = bbr.equality) 
            nll = LinkedRegressor(vars = bbr.vars, dependent_var = bbr.dependent_var)
            nll.mi_constraints = mic
            nll.leaf_variables = lv       
            push!(bbr.lls, nll)
        end
        bbr.active_trees[idx] = bbr.thresholds[idx]    
    else

        mi_constraints, leaf_variables = add_regr_constraints!(gm.model, bbr.vars, bbr.dependent_var, bbr.learners[idx], bbl, bbr.ul_data[idx];
                                                                equality = bbr.equality, lrs = bbr.lls)
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
    NOTE: mic and lv are only nonempty if we are adding an OCT approximation of a BBR. 
"""
function add_feas_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, lnr::Union{IAI.OptimalTreeLearner, AbstractModel}, bbl::Union{Nothing, BlackBoxLearner};
							   equality::Bool = false, 
							   relax_var::Union{Real, JuMP.VariableRef} = 0,
							   lcs::Array = [], mic::Dict = Dict(), lv::Dict = Dict(), ro_factor::Real = 0)
	#relax_var = 0
	gamma = 1
    constr_name = isnothing(bbl) ? "" : bbl.name 
	# if lnr isa AbstractModel
	#	 println(x)

	#	 lnr
	#	 β0, β = lnr.β0, lnr.β
	#	 @constraint(m, x'*β .+ 0.5 >=0)
	#	 return Dict(), Dict()
	# end
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
	mi_constraints = Dict(leaf => [] for leaf in feas_leaves)
	leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array}}()
	if isempty(lv)
		leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array}}()
		for leaf in feas_leaves
			leaf_variables[leaf], mi_constraints[leaf] = bounded_aux(x, @variable(m, binary=true))
		end
		mi_constraints[1] = [@constraint(m, sum(leaf_variables[leaf][1] for leaf in feas_leaves) == 1)]
		[push!(mi_constraints[1], constr) 
			for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in feas_leaves) .== x)]
		for lc in lcs
			for leaf in feas_leaves
				lc.leaf_variables[leaf], lc.mi_constraints[leaf] = bounded_aux(x, @variable(m, binary=true))
			end
			lc.mi_constraints[1] = [@constraint(m, sum(lc.leaf_variables[leaf][1] for leaf in feas_leaves) == 1)]
			[push!(lc.mi_constraints[1], constr) 
				for constr in @constraint(m, sum(lc.leaf_variables[leaf][2] for leaf in feas_leaves) .== lc.vars)]
			lc.active_leaves = []
		end
	else	
		leaf_variables = lv
	end
	if !isempty(mic)
		merge!(append!, mi_constraints, mic)
	end
	# Getting lnr data
	upperDict, lowerDict = trust_region_data(lnr, Symbol.(x))
	for i = 1:length(feas_leaves)
		leaf = feas_leaves[i]
		# ADDING TRUST REGIONS
		for (j, region) in enumerate(upperDict[leaf])
			threshold, α = region
			if ro_factor == 0
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= sum(α .* leaf_variables[leaf][2]) + relax_var))
				for lc in lcs
					push!(lc.mi_constraints[leaf], @constraint(m, threshold * lc.leaf_variables[leaf][1] <= sum(α .* lc.leaf_variables[leaf][2]) + 
													lc.relax_var))
				end
			else
				# (A_all_u, t_all_u, A_all_l, t_all_l) = bbl.ro_data
                
                #a_list, t_list = find_closest_planes(α, threshold, A_all_u, t_all_u)
                #mt = sum(t_list)/length(t_list)
                #threshold = maximum([threshold, (mt-threshold)*ro_factor/10])


				P = ro_factor*diagm(1.0*α)
				
				var_name = eval(Meta.parse(":t_ru_$(constr_name)_$(i)_$(j)"));
				m[var_name] = @variable(m, base_name=string(var_name));
				t_var = m[var_name];
				
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + gamma*t_var <= sum(α .* leaf_variables[leaf][2]) + relax_var))
				append!(mi_constraints[leaf], @constraint(m, P*leaf_variables[leaf][2] .<= t_var))
				append!(mi_constraints[leaf], @constraint(m, -P*leaf_variables[leaf][2] .<= t_var))

				for (k, lc) in enumerate(lcs)

					
					var_name = eval(Meta.parse(":t_ru_$(constr_name)_$(i)_$(j)_$(k)"));
					m[var_name] = @variable(m, base_name=string(var_name));
					t_var = m[var_name];
					
					push!(lc.mi_constraints[leaf], @constraint(m, threshold * lc.leaf_variables[leaf][1] + gamma*t_var <= sum(α .* lc.leaf_variables[leaf][2]) + 
								lc.relax_var))
					append!(lc.mi_constraints[leaf], @constraint(m, P*lc.leaf_variables[leaf][2] .<= t_var))
                    append!(lc.mi_constraints[leaf], @constraint(m, -P*lc.leaf_variables[leaf][2] .<= t_var))
				end
			end


		end
		for (j, region) in enumerate(lowerDict[leaf])
			threshold, α = region
			
			if ro_factor == 0	
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= sum(α .* leaf_variables[leaf][2])))
				for lc in lcs
					push!(lc.mi_constraints[leaf], @constraint(m, threshold  * lc.leaf_variables[leaf][1] + lc.relax_var >= 
																	sum(α .* lc.leaf_variables[leaf][2])))
				end
			else
                # (A_all_u, t_all_u, A_all_l, t_all_l) = bbl.ro_data
                
                # a_list, t_list = find_closest_planes(α, threshold, A_all_l, t_all_l)
                # mt = sum(t_list)/length(t_list)
                # threshold = minimum([threshold, (threshold-mt)*ro_factor/10])

				P = ro_factor*diagm(1.0*α)
				
				var_name = eval(Meta.parse(":t_rl_$(constr_name)_$(i)_$(j)"));
				m[var_name] = @variable(m, base_name=string(var_name));
				t_var = m[var_name];
				
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= gamma*t_var+sum(α .* leaf_variables[leaf][2])))
				append!(mi_constraints[leaf], @constraint(m, P*leaf_variables[leaf][2] .<= t_var))
				append!(mi_constraints[leaf], @constraint(m, -P*leaf_variables[leaf][2] .<= t_var))

				for (k, lc) in enumerate(lcs)
					
					var_name = eval(Meta.parse(":t_rl_$(constr_name)_$(i)_$(j)_$(k)"));
					m[var_name] = @variable(m, base_name=string(var_name));
					t_var = m[var_name];
					
					push!(lc.mi_constraints[leaf], @constraint(m, threshold  * lc.leaf_variables[leaf][1] + lc.relax_var >= 
																	sum(α .* lc.leaf_variables[leaf][2]))+ gamma*t_var)
					
					append!(lc.mi_constraints[leaf], @constraint(m, P*lc.leaf_variables[leaf][2] .<= t_var))
                    append!(lc.mi_constraints[leaf], @constraint(m, -P*lc.leaf_variables[leaf][2] .<= t_var))
				end
				
			end
		end
	end
	if equality
		infeas_leaves = [i for i in all_leaves if !Bool(IAI.get_classification_label(lnr, i))];
		for leaf in infeas_leaves
			leaf_variables[leaf], mi_constraints[leaf] = bounded_aux(x, @variable(m, binary=true))
		end
		push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][1] for leaf in infeas_leaves) == 1))
		[push!(mi_constraints[1], constr) 
			for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in infeas_leaves) .== x)]
		for lc in lcs
			for leaf in infeas_leaves
				lc.leaf_variables[leaf], lc.mi_constraints[leaf] = bounded_aux(x, @variable(m, binary=true))
			end
			push!(lc.mi_constraints[1], @constraint(m, sum(lc.leaf_variables[leaf][1] for leaf in infeas_leaves) == 1))
			[push!(lc.mi_constraints[1], constr) 
				for constr in @constraint(m, sum(lc.leaf_variables[leaf][2] for leaf in infeas_leaves) .== lc.vars)]
			lc.active_leaves = []
		end
		for i = 1:size(infeas_leaves, 1)
			leaf = infeas_leaves[i]
			# ADDING TRUST REGIONS
			for region in upperDict[leaf]
				threshold, α = region
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= sum(α .* leaf_variables[leaf][2]) + relax_var))
				for lc in lcs
					push!(lc.mi_constraints[leaf], @constraint(m, threshold * lc.leaf_variables[leaf][1] <= sum(α .* lc.leaf_variables[leaf][2]) + lc.relax_var))
				end
			end
			for region in lowerDict[leaf]
				threshold, α = region
				push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= sum(α .* leaf_variables[leaf][2])))
				for lc in lcs
					push!(lc.mi_constraints[leaf], @constraint(m, threshold * lc.leaf_variables[leaf][1] + lc.relax_var >= 
																	sum(α .* lc.leaf_variables[leaf][2])))
				end
			end
		end
	end
	return mi_constraints, leaf_variables
end
"""
    add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, lnr::IAI.OptimalTreeClassifier, 
            ul_data::Dict; equality::Bool = false)

Creates a set of MIO constraints from a OptimalTreeClassifier that thresholds a BlackBoxRegressor.
Arguments:
    m:: JuMP Model
    x:: independent JuMPVariable (features in lnr)
    y:: dependent JuMPVariable (output of lnr)
    lnr:: A fitted OptimalTreeRegressor
    ul_data:: Upper and lower bounding hyperplanes for data in leaves of lnr
"""
function add_regr_constraints!(m::JuMP.Model, x::Array{JuMP.VariableRef}, y::JuMP.VariableRef, 
                               lnr::IAI.OptimalTreeLearner, bbl::Union{Nothing, BlackBoxLearner},
                               ul_data::Dict; equality::Bool = false, 
                               relax_var::Union{Real, JuMP.VariableRef} = 0,
                               lrs::Array = [])
    # TODO: Determine whether I should relax approximator or trust regions or both. 
    if lnr isa OptimalTreeRegressor                
        check_if_trained(lnr)
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        mi_constraints = Dict(leaf => [] for leaf in all_leaves)
        leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}()
        for leaf in all_leaves
            leaf_variables[leaf], mi_constraints[leaf] = bounded_aux(x, y, @variable(m, binary=true))
        end
        mi_constraints[1] = [@constraint(m, sum(leaf_variables[leaf][1] for leaf in all_leaves) == 1)]
        [push!(mi_constraints[1], constr) 
            for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in all_leaves) .== x)]
        push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][3] for leaf in all_leaves) == y))
        for lr in lrs          
            for leaf in all_leaves
                lr.leaf_variables[leaf], lr.mi_constraints[leaf] = bounded_aux(lr.vars, lr.dependent_var, @variable(m, binary=true))
            end
            lr.mi_constraints[1] = [@constraint(m, sum(lr.leaf_variables[leaf][1] for leaf in all_leaves) == 1)]
            [push!(lr.mi_constraints[1], constr)
                for constr in @constraint(m, sum(lr.leaf_variables[leaf][2] for leaf in all_leaves) .== lr.vars)]
            push!(lr.mi_constraints[1], @constraint(m, sum(lr.leaf_variables[leaf][3] for leaf in all_leaves) == lr.dependent_var))
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
                push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] + relax_var >= leaf_variables[leaf][3]))
                [push!(lr.mi_constraints[leaf],  @constraint(m, sum(β .* lr.leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] + lr.relax_var >= lr.leaf_variables[leaf][3])) for lr in lrs]
                push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.leaf_variables[leaf][2]) + 
                    β0 * lr.leaf_variables[leaf][1] <= lr.leaf_variables[leaf][3] + lr.relax_var)) for lr in lrs]
            elseif !isempty(ul_data) 
                # Use only the lower approximator
                β0, β = ul_data[leaf] # update the lower approximator
                push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* lr.leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] <= lr.leaf_variables[leaf][3] + lr.relax_var)) for lr in lrs]
            else 
                # Use only the ridge regressor as the lower approximator. 
                push!(mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] <= leaf_variables[leaf][3] + relax_var))
                [push!(lr.mi_constraints[leaf], @constraint(m, sum(β .* leaf_variables[leaf][2]) + 
                    β0 * leaf_variables[leaf][1] <= lr.leaf_variables[leaf][3] + lr.relax_var)) for lr in lrs]
            end
            # ADDING TRUST REGIONS
            for region in upperDict[leaf]
                threshold, α = region
                push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] <= 
                    sum(α .* leaf_variables[leaf][2]) + relax_var))
                [push!(lr.mi_constraints[leaf], @constraint(m, threshold * lr.leaf_variables[leaf][1] <= 
                    sum(α .* lr.leaf_variables[leaf][2]) + lr.relax_var)) for lr in lrs]
            end
            for region in lowerDict[leaf]
                threshold, α = region
                push!(mi_constraints[leaf], @constraint(m, threshold * leaf_variables[leaf][1] + relax_var >= 
                sum(α .* leaf_variables[leaf][2])))
                [push!(lr.mi_constraints[leaf], @constraint(m, threshold * lr.leaf_variables[leaf][1] + 
                    relax_var >= sum(α .* lr.leaf_variables[leaf][2]))) for lr in lrs]
            end
        end
        return mi_constraints, leaf_variables
    elseif lnr isa OptimalTreeClassifier
        isempty(lrs) || throw(OCTHaGOnException("Bug: Cannot use OCTs to approximate linked BBRs."))
        # Add a binary variable for each leaf
        all_leaves = find_leaves(lnr)
        # Add a binary variable for each leaf
        feas_leaves = [i for i in all_leaves if Bool(IAI.get_classification_label(lnr, i))]
        mi_constraints = Dict(leaf => [] for leaf in feas_leaves)
        leaf_variables = Dict{Int64, Tuple{JuMP.VariableRef, Array, JuMP.VariableRef}}()
        for leaf in feas_leaves
            leaf_variables[leaf], mi_constraints[leaf] = bounded_aux(x, y, @variable(m, binary=true))
        end
        mi_constraints[1] = [@constraint(m, sum(leaf_variables[leaf][1] for leaf in feas_leaves) == 1)]
        [push!(mi_constraints[1], constr) 
            for constr in @constraint(m, sum(leaf_variables[leaf][2] for leaf in feas_leaves) .== x)]
        push!(mi_constraints[1], @constraint(m, sum(leaf_variables[leaf][3] for leaf in feas_leaves) == y))
        mi_constraints, leaf_variables = add_feas_constraints!(m, x, lnr, bbl, equality = equality, lcs = lrs, 
                                                    mic = mi_constraints, lv = leaf_variables)
        if !isempty(ul_data)
            if all(keys(ul_data) .<= 0) || !all(keys(ul_data) .>= 0) # means an upper or upperlower bounding tree
                mi_constraints = Dict(-key => value for (key, value) in mi_constraints) # hacky sign flipping for upkeep. 
                leaf_variables = Dict(-key => value for (key, value) in leaf_variables) 
            end
            for leaf in collect(keys(ul_data))
                γ0, γ = ul_data[leaf]
                if !haskey(mi_constraints, leaf) # occurs with ORTS with bounding hyperplanes
                    mi_constraints[leaf] = [@constraint(m, leaf_variables[-leaf][3] <= γ0 * leaf_variables[-leaf][1] + 
                        sum(γ .* leaf_variables[-leaf][2]) + relax_var)]
                elseif leaf <= 0
                    push!(mi_constraints[leaf], @constraint(m, leaf_variables[leaf][3] <= γ0 * leaf_variables[leaf][1] + 
                        sum(γ .* leaf_variables[leaf][2]) + relax_var))
                else
                    push!(mi_constraints[leaf], @constraint(m, leaf_variables[leaf][3] + relax_var >= 
                    γ0 * leaf_variables[leaf][1] + sum(γ .* leaf_variables[leaf][2])))
                end
            end
        end
        return mi_constraints, leaf_variables
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
                    throw(OCTHaGOnException("Bug: Constraints could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, (bin_var, leaf_vars, aux_dep)) in bbr.leaf_variables
        if leaf_key <= 0
            for leaf_var in leaf_vars
                if is_valid(gm.model, leaf_var)
                    delete(gm.model, leaf_var)
                else
                    throw(OCTHaGOnException("Bug: Variables could not be removed."))
                end
            end
            if is_valid(gm.model, bin_var)
                delete(gm.model, bin_var)
            else
                throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
            end
            if is_valid(gm.model, aux_dep)
                delete(gm.model, aux_dep)
            else
                throw(OCTHaGOnException("Bug: Aux dependent variable could not be removed."))
            end
            delete!(bbr.leaf_variables, leaf_key)
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
                    throw(OCTHaGOnException("Bug: Constraints could not be removed."))
                end
            end
            delete!(bbr.mi_constraints, leaf_key)
        end
    end
    for (leaf_key, (bin_var, leaf_vars, aux_dep)) in bbr.leaf_variables
        if leaf_key >= 0
            for leaf_var in leaf_vars
                if is_valid(gm.model, leaf_var)
                    delete(gm.model, leaf_var)
                else
                    throw(OCTHaGOnException("Bug: Variables could not be removed."))
                end
            end
            if is_valid(gm.model, bin_var)
                delete(gm.model, bin_var)
            else
                throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
            end
            if is_valid(gm.model, aux_dep)
                delete(gm.model, aux_dep)
            else
                throw(OCTHaGOnException("Bug: Aux dependent variable could not be removed."))
            end
            delete!(bbr.leaf_variables, leaf_key)
        end
    end
    bbr.active_leaves = []
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
                throw(OCTHaGOnException("Bug: Constraints could not be removed."))
            end
        end
        delete!(bbc.mi_constraints, leaf_key)
    end
    for (leaf_key, (bin_var, leaf_vars)) in bbc.leaf_variables
        for leaf_var in leaf_vars
            if is_valid(gm.model, leaf_var)
                delete(gm.model, leaf_var)
            else
                throw(OCTHaGOnException("Bug: Variables could not be removed."))
            end
        end
        if is_valid(gm.model, bin_var)
            delete(gm.model, bin_var)
        else
            throw(OCTHaGOnException("Bug: Binary variable could not be removed."))
        end
        delete!(bbc.leaf_variables, leaf_key)
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
            throw(OCTHaGOnException("Hypertype $(hypertype) not recognized."))
        end
        add_tree_constraints!(gm, bbr, idx)
        return 
    else
        throw(OCTHaGOnException("$(gm.name) has too many active lower/upper bounding regressors."))
    end
end

function update_tree_constraints!(gm::GlobalModel, bbc::BlackBoxClassifier, idx = length(bbc.learners))
    clear_tree_constraints!(gm, bbc)
    add_tree_constraints!(gm, bbc, idx)
    return
end
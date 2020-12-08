"""
Contains all required info to be able to generate a global optimization problem.
NOTE: proper construction is to use add_nonlinear_constraint to add bbfs.
model must be a mixed integer convex model.
nonlinear_model can contain JuMP.NonlinearConstraints.
"""
@with_kw mutable struct GlobalModel
    model::JuMP.Model                                                     # JuMP model
    name::Union{Symbol, String} = "Model"                                 # Example name
    bbfs::Array{BlackBoxFunction} = Array{BlackBoxFunction}[]              # Black box (>/= 0) functions
    vars::Array{VariableRef} = JuMP.all_variables(model)                  # JuMP variables
end

"""
    (gm::GlobalModel)(name::Union{String, Int64})

Finds BlackBoxFunction in GlobalModel by name.
"""
function (gm::GlobalModel)(name::Union{String, Int64})
    fn_names = getfield.(gm.bbfs, :name)
    fns = gm.bbfs[findall(x -> x == name, fn_names)]
    if length(fns) == 1
        return fns[1]
    elseif length(fns) == 0
        @warn("No constraint with name ", name)
        return
    else
        @warn("Multiple constraints with name ", name)
        return fns
    end
end


"""
    JuMP.all_variables(gm::Union{GlobalModel, BlackBoxFunction})

Extends JuMP.all_variables to GMs and BBFs
and makes sure that the variables are updated.
"""
function JuMP.all_variables(gm::Union{GlobalModel, BlackBoxFunction})
    if gm isa GlobalModel
        gm.vars = JuMP.all_variables(gm.model)
    end
    return gm.vars
end

function JuMP.set_optimizer(gm::GlobalModel, optimizer_factory)
    """ Extends JuMP.set_optimizer to GlobalModels. """
    set_optimizer(gm.model, optimizer_factory)
end

function JuMP.optimize!(gm::GlobalModel)
    """ Extends JuMP.optimize! to GlobalModels. """
    optimize!(gm.model)
end

function get_bounds(model::Union{GlobalModel, JuMP.Model, BlackBoxFunction})
    """ Returns bounds of all variables from a JuMP.Model. """
    all_vars = all_variables(model)
    return get_bounds(all_vars)
end

function get_bounds(bbfs::Array{BlackBoxFunction})
    """ Returns bounds of all variables from a JuMP.Model. """
    all_vars = unique(Iterators.flatten(([all_variables(bbf) for bbf in bbfs])))
    return get_bounds(all_vars)
end

function check_bounds(bounds::Dict)
    """ Checks outer-boundedness. """
    if any(isinf.(Iterators.flatten(values(bounds))))
        throw(OCTException("Unbounded variables in model!"))
    else
        return
    end
end

function get_max(a, b)
    return maximum([a,b])
end

function get_min(a,b)
    return minimum([a,b])
end

function check_infeasible_bounds(model::Union{GlobalModel, JuMP.Model}, bounds::Dict)
    all_bounds = get_bounds(model);
    lbs_model = Dict(key => minimum(value) for (key, value) in all_bounds)
    ubs_model = Dict(key => maximum(value) for (key, value) in all_bounds)
    lbs_bounds = Dict(key => minimum(value) for (key, value) in bounds)
    ubs_bounds = Dict(key => maximum(value) for (key, value) in bounds)
    nlbs = merge(get_max, lbs_model, lbs_bounds)
    nubs = merge(get_min, ubs_model, ubs_bounds)
    if any([nlbs[var] .> nubs[var] for var in keys(nlbs)])
        throw(OCTException("Infeasible bounds."))
    end
    return
end

function add_nonlinear_constraint(gm::GlobalModel,
                     constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     name::Union{Nothing, String} = nothing,
                     equality::Bool = false)
""" Adds a new nonlinear constraint to Global Model.
    Note: Linear constraints should be added using add_linear_constraint. """
    bbf_vars = []
    if isnothing(vars)
        bbf_vars = JuMP.all_variables(gm.model)
    else
        bbf_vars = vars
    end
    if isnothing(name)
        name = string("bbf", length(gm.bbfs) + 1)
    end
    if isnothing(expr_vars) && constraint isa Expr
        expr_vars = vars_from_expr(constraint, vars[1].model)
    end
    if constraint isa JuMP.ScalarConstraint #TODO: clean up.
        con = JuMP.add_constraint(gm.model, bbf.constraint)
        JuMP.delete(gm.model, con)
        new_bbf = BlackBoxFunction(constraint = con, vars = bbf_vars, expr_vars = expr_vars,
                                   equality = equality, name = name)
        @assert length(new_bbf.vars) == length(new_bbf.varmap)
        push!(gm.bbfs, new_bbf)
        return
    end
    if constraint isa JuMP.ConstraintRef
        JuMP.delete(gm.model, constraint)
    end
    new_bbf = BlackBoxFunction(constraint = constraint, vars = bbf_vars, expr_vars = expr_vars,
                               equality = equality, name = name)
    @assert length(new_bbf.vars) == length(new_bbf.varmap)
    push!(gm.bbfs, new_bbf)
    return
end

"""
    add_nonlinear_or_compatible(gm::GlobalModel,
                         constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                         vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                         expr_vars::Union{Nothing, Array} = nothing,
                         name::Union{Nothing, String} = nothing,
                         equality::Bool = false)

Extents add_nonlinear_constraint to recognize JuMP compatible constraints and add them
as normal JuMP constraints
"""
function add_nonlinear_or_compatible(gm::GlobalModel,
                     constraint::Union{JuMP.ScalarConstraint, JuMP.ConstraintRef, Expr};
                     vars::Union{Nothing, Array{JuMP.VariableRef, 1}} = nothing,
                     expr_vars::Union{Nothing, Array} = nothing,
                     name::Union{Nothing, String} = nothing,
                     equality::Bool = false)
     fn = eval(constraint)
     @assert fn isa Function
     try
        constr_expr = fn(expr_vars...)
        if equality
            @constraint(model, constr_expr == 0)
        else
            @constraint(model, constr_expr >= 0)
        end
     catch
         add_nonlinear_constraint(gm, constraint, vars = vars, expr_vars = expr_vars,
                                      equality = equality, name = name)
     end
end

"""
    nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxFunction})
    nonlinearize!(gm::GlobalModel)

Turns gm.model into the nonlinear representation.
NOTE: to get back to MI-compatible forms, must rebuild model from scratch.
"""
function nonlinearize!(gm::GlobalModel, bbfs::Array{BlackBoxFunction})
    for (i, bbf) in enumerate(bbfs)
        if bbf.constraint isa JuMP.ConstraintRef
            JuMP.add_constraint(gm.model, bbf.constraint)
        elseif bbf.constraint isa Expr
            symb = Symbol(bbf.name)
            expr_vars = bbf.expr_vars
            vars = flat(expr_vars)
            var_ranges = []
            count = 0
            for varlist in expr_vars
                if varlist isa VariableRef
                    count += 1
                    push!(var_ranges, count)
                else
                    push!(var_ranges, (count + 1 : count + length(varlist)))
                    count += length(varlist)
                end
            end
            expr = bbf.constraint
            flat_expr = :((x...) -> $(expr)([x[i] for i in $(var_ranges)]...))
            fn = eval(flat_expr)
            JuMP.register(gm.model, symb, length(vars), fn; autodiff = true)
            expr = Expr(:call, symb, vars...)
            if bbf.equality
                JuMP.add_NL_constraint(gm.model, :($(expr) == 0))
            else
                JuMP.add_NL_constraint(gm.model, :($(expr) >= 0))
            end
        end
    end
    return
end

function nonlinearize!(gm::GlobalModel)
    nonlinearize!(gm, gm.bbfs)
end

function bound!(model::Union{GlobalModel, JuMP.Model},
                bounds::Dict)
    """Adds outer bounds to JuMP Model from dictionary of data. """
    if model isa GlobalModel
        bound!(model.model, bounds)
        return
    end
    check_infeasible_bounds(model, bounds)
    for (key, value) in bounds
        @assert value isa Array && length(value) == 2
        var = fetch_variable(model, key);
        if var isa Array # make sure all elements are bounded.
            for v in var
                bound!(model, Dict(v => value))
            end
        else
            if JuMP.has_lower_bound(var) && JuMP.lower_bound(var) <= minimum(value)
                set_lower_bound(var, minimum(value))
            elseif !JuMP.has_lower_bound(var)
                set_lower_bound(var, minimum(value))
            end
            if JuMP.has_upper_bound(var) && JuMP.upper_bound(var) >= maximum(value)
                set_upper_bound(var, maximum(value))
            else !JuMP.has_upper_bound(var)
                set_upper_bound(var, maximum(value))
            end
        end
    end
    return
end

function classify_constraints(model::Union{GlobalModel, JuMP.Model})
    """Separates and returns linear and nonlinear constraints in a model. """
    jump_model = model
    if model isa GlobalModel
        jump_model = model.model
    end
    all_types = list_of_constraint_types(jump_model)
    nl_constrs = [];
    l_constrs = [];
    l_vartypes = [JuMP.VariableRef, JuMP.GenericAffExpr{Float64, VariableRef}]
    l_constypes = [MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64}]
    for (vartype, constype) in all_types
        constrs_of_type = JuMP.all_constraints(jump_model, vartype, constype)
        if any(vartype .== l_vartypes) && any(constype .== l_constypes)
            append!(l_constrs, constrs_of_type)
        else
            append!(nl_constrs, constrs_of_type)
        end
    end
    if !isnothing(jump_model.nlp_data)
        append!(nl_constrs, jump_model.nlp_data.nlconstr)
    end
    return l_constrs, nl_constrs
end

""" Returns the feasibility of data points in a BBF or GM. """
function feasibility(bbf::Union{GlobalModel, Array{BlackBoxFunction, DataConstraint}, DataConstraint, BlackBoxFunction})
    if isa(bbf, Union{BlackBoxFunction, DataConstraint})
        return bbf.feas_ratio
    elseif isa(bbf, Array{BlackBoxFunction, DataConstraint})
        return [feasibility(fn) for fn in bbf]
    else
        return [feasibility(fn) for fn in bbf.bbfs]
    end
end

""" Returns the accuracy of learners in a BBF or GM. """
function accuracy(bbf::Union{GlobalModel, BlackBoxFunction})
    if isa(bbf, BlackBoxFunction)
        if bbf.feas_ratio in [1., 0]
            @warn(string("Accuracy of BlackBoxFunction ", bbf.name, " is tautological."))
            return 1.
        elseif isempty(bbf.learners)
            throw(OCTException(string("BlackBoxFunction ", bbf.name, " has not been trained yet.")))
        else
            return bbf.accuracies[end]
        end
    else
        return [accuracy(fn) for fn in bbf.bbfs]
    end
end

function lh_sample(bbf::BlackBoxFunction; iterations::Int64 = 3,
                n_samples::Int64 = 1000)
"""
Uniformly Latin Hypercube samples the variables of GlobalModel, as long as all
lbs and ubs are defined.
"""
   bounds = get_bounds(bbf.vars)
   check_bounds(bounds)
   n_dims = length(bbf.vars)
   plan, _ = LHCoptim(n_samples, n_dims, iterations);
   X = scaleLHC(plan,[(minimum(bounds[var]), maximum(bounds[var])) for var in bbf.vars]);
   return DataFrame(X, string.(bbf.vars))
end

function choose(large::Int64, small::Int64)
    return Int64(factorial(big(large)) / (factorial(big(large-small))*factorial(big(small))))
end

function boundary_sample(bbf::BlackBoxFunction; fraction::Float64 = 0.5)
""" *Smartly* samples the constraint along the variable boundaries.
    NOTE: Because we are sampling symmetrically for lower and upper bounds,
    the choose coefficient has to be less than ceil(half of number of dims). """
    bounds = get_bounds(bbf.vars);
    check_bounds(bounds);
    n_vars = length(bbf.vars);
    vks = string.(bbf.vars);
    lbs = DataFrame(Dict(string(key) => minimum(value) for (key, value) in bounds))
    ubs = DataFrame(Dict(string(key) => maximum(value) for (key, value) in bounds))
    n_comb = sum(choose(n_vars, i) for i=0:n_vars);
    nX = DataFrame([Float64 for i in vks], vks)
    sample_indices = [];
    if n_comb >= fraction*bbf.n_samples
        @warn("Can't exhaustively sample the boundary of Constraint " * string(bbf.name) * ".")
        n_comb = 2*n_vars+2; # Everything is double because we choose min's and max's
        choosing = 1;
        while n_comb <= fraction*bbf.n_samples
            choosing = choosing + 1;
            n_comb += 2*choose(n_vars, choosing);
        end
        choosing = choosing - 1; # Determined maximum 'choose' coefficient
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:choosing); # Choose 1 and above
    else
        sample_indices = reduce(vcat,collect(combinations(1:n_vars,i)) for i=0:n_vars); # Choose 1 and above
    end
    for i in sample_indices
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        lbscopy[:, vks[i]] = ubscopy[:, vks[i]];
        append!(nX, lbscopy);
        lbscopy = copy(lbs); ubscopy = copy(ubs);
        ubscopy[:, vks[i]] = lbscopy[:, vks[i]];
        append!(nX, ubscopy);
    end
    return nX
end

function knn_sample(bbf::BlackBoxFunction; k::Int64 = 15)
    """ Does KNN and interval arithmetic based sampling once there is at least one feasible
        sample to a BlackBoxFunction. """
    if bbf.feas_ratio == 0. || bbf.feas_ratio == 1.0
        throw(OCTException("Constraint " * string(bbf.name) * " must have at least one feasible or
                            infeasible sample to be KNN-sampled!"))
    end
    vks = string.(bbf.vars)
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbf);
    idxs, dists = find_knn(bbf, k=k);
    positives = findall(x -> x .>= 0 , bbf.Y);
    feas_class = classify_patches(bbf, idxs);
    for center_node in positives # This loop is for making sure that every possible root is sampled only once.
        if feas_class[center_node] == "mixed"
            nodes = [idx for idx in idxs[center_node] if bbf.Y[idx] < 0];
            push!(nodes, center_node)
            np = secant_method(bbf.X[nodes, :], bbf.Y[nodes, :])
            append!(df, np);
        end
    end
    return df
end

function sample_and_eval!(bbf::Union{BlackBoxFunction, GlobalModel, Array{BlackBoxFunction}};
                          n_samples:: Union{Int64, Nothing} = nothing,
                          boundary_fraction::Float64 = 0.5,
                          iterations::Int64 = 3)
    """ Samples and evaluates BlackBoxFunction, with n_samples new samples.
    Arguments
    n_samples: number of samples, overwrites bbf.n_samples.
    boundary_fraction: maximum ratio of boundary samples
    iterations: number of GA populations for LHC sampling (0 is a random LH.)
    ratio:
    If there is an optimized gp, ratio*n_samples is how many random LHC samples are generated
    for prediction from GP. """
    if bbf isa GlobalModel
        for fn in bbf.bbfs
            sample_and_eval!(fn, n_samples = n_samples, boundary_fraction = boundary_fraction,
                             iterations = iterations)
        end
        return
    elseif bbf isa Array{BlackBoxFunction}
        for fn in bbf
            sample_and_eval!(fn, n_samples = n_samples, boundary_fraction = boundary_fraction,
                             iterations = iterations)
        end
        return
    end
    if !isnothing(n_samples)
        bbf.n_samples = n_samples
    end
    vks = string.(bbf.vars)
    n_dims = length(vks);
    check_bounds(get_bounds(bbf))
    if size(bbf.X,1) == 0 # If we don't have data yet, uniform and boundary sample.
       df = boundary_sample(bbf, fraction = boundary_fraction)
       eval!(bbf, df)
       df = lh_sample(bbf, iterations = iterations, n_samples = bbf.n_samples - size(df, 1))
       eval!(bbf, df);
    elseif bbf.feas_ratio == 1.0
        @warn(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.")
    elseif bbf.feas_ratio == 0.0
        throw(OCTException(string(bbf.name) * " was not KNN sampled since it has " * string(bbf.feas_ratio) * " feasibility.
                            Please find at least one feasible sample and try again. "))
    else                  # otherwise, KNN sample!
        df = knn_sample(bbf)
        eval!(bbf, df)
    end
    return
end

""" Extends JuMP.optimize! to GlobalModels. """
function JuMP.optimize!(gm::GlobalModel; kwargs...)
    optimize!(gm.model, kwargs...)
end

function globalsolve(gm::GlobalModel)
    """ Creates and solves the global optimization model using the linear constraints from GlobalModel,
        and approximated nonlinear constraints from inside its BlackBoxFunctions."""
    clear_tree_constraints!(gm); # remove trees from previous solve (if any).
    add_tree_constraints!(gm); # refresh latest tree constraints.
    status = JuMP.optimize!(gm.model);
    return status
end

""" Returns the optimal solution of the solved global optimization model. """
function solution(gm::GlobalModel)
    vals = getvalue.(gm.vars)
    return DataFrame(vals', string.(gm.vars))
end

function evaluate_feasibility(gm::GlobalModel)
    """ Evaluates each constraint at solution to make sure it is feasible. """
    soln = solution(gm);
    feas = [];
    for fn in gm.bbfs
        eval!(fn, soln)
    end
    infeas_idxs = findall(idx -> gm.bbfs[idx].Y[end] .< 0, collect(1:length(gm.bbfs))) #TODO: optimize
    feas_idxs = findall(idx -> gm.bbfs[idx].Y[end] .>= 0, collect(1:length(gm.bbfs)))
    return feas_idxs, infeas_idxs
end

"""
    find_bounds!(gm::GlobalModel; bbfs::Array{BlackBoxFunction} = [], M = 1e5, all_bounds::Bool=true)

Finds the outer variable bounds of GlobalModel by solving only over the linear constraints
and listed BBFs.
TODO: improve! Only find bounds of non-binary variables.
"""
function find_bounds!(gm::GlobalModel; bbfs::Array{BlackBoxFunction} = BlackBoxFunction[], M = 1e5, all_bounds::Bool=true)
    new_bounds = Dict(var => [-Inf, Inf] for var in gm.vars)
    # Finding bounds by min/maximizing each variable
    clear_tree_constraints!(gm)
    if !isempty(bbfs)
        add_tree_constraints!(gm, bbfs, M=M)
    end
    m = gm.model;
    x = gm.vars;
    old_bounds = get_bounds(m);
    orig_objective = JuMP.objective_function(gm.model)
    @showprogress 0.5 "Finding bounds..." for var in gm.vars
        if isinf(old_bounds[var][1]) || all_bounds
            @objective(m, Min, var);
            JuMP.optimize!(m);
            new_bounds[var][1] = getvalue(var);
        end
        if isinf(old_bounds[var][2]) || all_bounds
            @objective(m, Max, var);
            JuMP.optimize!(m);
            new_bounds[var][2] = getvalue(var);
        end
    end
    # Revert objective
    @objective(m, Min, orig_objective)
    bound!(gm, new_bounds)
    return
end

function import_trees(dir, gm::GlobalModel)
    """ Returns trees trained over given GlobalModel,
    where filename points to the model name. """
    trees = [IAI.read_json(string(dir, "_tree_", i, ".json")) for i=1:length(gm.bbfs)];
    return trees
end

function clear_data!(bbf::BlackBoxFunction)
    bbf.X = DataFrame([Float64 for i=1:length(bbf.vars)], string.(bbf.vars))  # Function samples
    bbf.Y = [];
    return
end

function clear_data!(gm::GlobalModel)
    clear_data!.(gm.bbfs)
    return
end
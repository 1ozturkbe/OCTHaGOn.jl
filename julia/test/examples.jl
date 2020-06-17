using ConicBenchmarkUtilities
using LatinHypercubeSampling
using MathOptInterface, MosekTools
const MOI = MathOptInterface
using MathProgBase
using PyCall
using Test

include("../src/structs.jl");
include("../src/fit.jl");

function example_fit(md::ModelData; lnr=base_otc())
    """ Fits a provided function model with feasibility and obj f'n fits and
        saves the learners.
    """
    n_samples = 1000;
    n_dims = length(md.lbs);
    weights = ones(n_samples)
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(md.lbs[i], md.ubs[i]) for i=1:n_dims]);
    ineq_trees = learn_constraints!(md, md.ineq_fns, X, weights=weights, lnr=lnr)
    for i=1:size(ineq_trees,1)
        IAI.write_json(string("data/", md.name, "_ineq_", i, ".json"),
                       ineq_trees[i])
    end
    return true
end

function example_solve(md::ModelData; M=1e5)
    """ Solves an already fitted function_model. """
    # Retrieving constraints
    obj_tree = IAI.read_json(string("data/", md.name, "_obj.json"));
    ineq_trees = [IAI.read_json(string("data/", md.name, "_ineq_", i, ".json")) for i=1:length(md.ineqs)];
    # Creating JuMP model
    m = Model(solver=GurobiSolver());
    n_vars = length(md.lbs);
    vks = [Symbol("x",i) for i=1:n_vars];
    obj_vks = [Symbol("x",i) for i=1:n_vars+1];
    @variable(m, x[1:n_vars])
    @variable(m, y)
    @objective(m, Min, y)
    add_feas_constraints!(obj_tree, m, [x; y], obj_vks; M=M)
    for tree in ineq_trees
        add_feas_constraints!(tree, m, x, vks; M=M)
    end
    for tree in eq_trees
        add_feas_constraints!(tree, m, x, vks; M=M)
    end
    for i=1:n_vars # Bounding
        @constraint(m, x[i] <= md.ubs[i])
        @constraint(m, x[i] >= md.lbs[i])
    end
    status = solve(m)
    return true
end

function alphac_to_fn(alpha, c; lse=false)
    n_terms, n_vars = size(alpha)
    if lse
        x -> sum([c[i]*exp(sum([alpha[i,j]*x[j] for j=1:n_vars])) for i=1:n_terms])
    else
        x -> sum([c[i]*prod([x[j]^alpha[i,j] for j=1:n_vars]) for i=1:n_terms])
    end
end

function import_sagebenchmark(idx; lse=false)
    """
    Imports sagebenchmarks example from literature.solved and
    returns as a function_model.
    Arguments:
        idx:: number of benchmark
    Returns:
        ex:: a function_model
    """
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(idx);
    f, greaters, equals = signomials;
    n_vars = size(f.alpha,2)+1;
    ineq_fns = Function[];
    ineq_idxs = Array[];
    eq_fns = Function[];
    eq_idxs = Array[];
    # Turning objective into constraint
    c = zeros(n_vars); c[end] = 1;
    obj_c = vcat(-1 .* f.c, [1]);
    obj_alpha = hcat(-1 .* f.alpha, zeros(length(c)-1));
    obj_alpha = vcat(obj_alpha, c');
    push!(ineq_fns, alphac_to_fn(obj_alpha, obj_c, lse=lse))
    push!(ineq_idxs, unique([idx[2] for idx in findall(i->i != 0, obj_alpha)]));
    # Bound initialization
    lbs = -Inf*ones(n_vars);
    ubs = Inf*ones(n_vars);
    for i=1:size(greaters,1)
        alpha = hcat(greaters[i].alpha, zeros(length(c)-1))
        c = hcat(greaters[i].c, [0])
        local idxs = findall(x->x!=0, alpha);
        if size(idxs, 1) > 1
            append!(ineq_fns, [alphac_to_fn(alpha, c; lse=lse)])
            append!(ineq_idxs, [unique([idx[2] for idx in idxs])]);
        else
            local val = -((sum(c)-c[idxs[1][1]]) / c[idxs[1][1]])^(1/alpha[idxs[1]]);
            if lse
                val=log(val)
            end
            if c[idxs[1][1]] <= 0 && (ismissing(ubs[idxs[1][2]]) || ubs[idxs[1][2]] >= val)
                ubs[idxs[1][2]] = val;
            elseif ismissing(lbs[idxs[1][2]]) || lbs[idxs[1][2]] <= val
                lbs[idxs[1][2]] = val;
            end
        end
    end
    for i=1:size(equals, 1)
        alpha = hcat(equals[i].alpha, zeros(length(c)-1))
        c = hcat(equals[i].c, [0]);
        idxs = findall(x->x!=0, alpha);
        append!(eqs, [alphac_to_fn(alpha, c; lse=lse)]);
    end
    ex = ModelData(name = string("example", idx),
                   obj_c = obj_c,
                   ineqs = ineqs, ineq_idxs = ineq_idxs,
                   eqs = eqs, eq_idxs, eq_idxs,
                   lbs = lbs, ubs = ubs);
    return ex
end

function CBF_to_MOF(filename, solver=Mosek.Optimizer())
    """ Imports a conic benchmark into a MathOptInterface format. """
    model = MOI.FileFormats.Model(filename = filename);
    mof_model = MOI.Bridges.full_bridge_optimizer(solver , Float64)
    MOI.read_from_file(model, filename)
    MOI.copy_to(mof_model, model)
    return mof_model
end

# function MOF_to_fnmodel(mof_model)
#     constraints = MOI.get(mof_model, MOI.ListOfConstraints())
#     constraint_indices = MOI.get(mof_model, MOI.ListOfConstraintIndices)
#     variable_attributes_set = MOI.get(mof_model, MOI.ListOfVariableAttributesSet())
#     #TODO: CONTINUE HERE
#     obj = model.objective.constant*model.objective.terms
#     lbs = model.lower_bounds;
#     ubs = model.upper_bounds;
#     ex = function_model(string("example", idx),
#                         obj, obj_idxs, ineqs, ineq_idxs,
#                         eqs, eq_idxs, lbs, ubs, lse);
#     return ex
# end

function CBF_to_ModelData(filename)
    """ Converts CBF model into a fnmodel format.
        Using some code here from MathProgBase loadproblem! (technically deprecated). """
    dat = readcbfdata(filename);
    c, A, b, constr_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat);
    # In MathProgBase format:
    # Conic form
    #  min  c'x
    #  st b-Ax ∈ K_1
    #        x ∈ K_2
    # Note: The sense in MathProgBase form is always minimization, and the objective offset is zero.
    # If sense == :Max, you should flip the sign of c before handing off to a solver.
    # If a cone is anything other than [:Free,:Zero,:NonNeg,:NonPos,:SOC,:SOCRotated], MPB
    # would give up. But we should be able to handle it.
    bad_cones = [:SDP, :ExpPrimal, :ExpDual];
    for (cone,idxs) in var_cones
        cone in bad_cones && error("Cone type $(cone) not supported")
    end
    n_vars = length(c);
    md = ModelData(c=c);
    # Variable bounds
    l = fill(-Inf, n_vars);
    u = fill(Inf, n_vars);
    for (cone,idxs) in var_cones
        if cone == :SOC
            l[idxs[1]] = 0;
        elseif cone == :SOCRotated
            l[idxs[1]] = 0;
            l[idxs[2]] = 0;
        else
            cone_l = (cone == :Free || cone == :NonPos) ? -Inf : 0.0;
            cone_u = (cone == :Free || cone == :NonNeg) ?  Inf : 0.0;
            for idx in idxs
                l[idx] = cone_l;
                u[idx] = cone_u;
            end
        end
    end
    # Constraint functions
    for (cone, idxs) in constr_cones
    # Idxs describe which rows of A the cone applies to.
    # All cones: [:Free, :Zero, :NonNeg, :NonPos, :SOC, :SOCRotated, :SDP, :ExpPrimal, :ExpDual]
        if cone == :NonNeg
            append!(md.ineqs_b, b[idxs]);
            push!(md.ineqs_A, A[idxs, :]);
            if length(A[idxs,:].nzval) == 1.
                l[findall(!iszero, A[idxs,:])[1][2]] = maximum([l[findall(!iszero, A[idxs,:])[1][2]],
                                                                b[idxs][1]/A[idxs, :].nzval[1]]);
            end
        elseif cone == :NonPos
            append!(md.ineqs_b, -b[idxs])
            push!(md.ineqs_A, -1 .* A[idxs, :])
            if length(idxs) == 1.
                u[findall(!iszero, A[idxs,:])[1][2]] = minimum([u[findall(!iszero, A[idxs,:])[1][2]],
                                                   b[idxs][1]/A[idxs, :].nzval[1]]);
            end
        elseif cone == :Zero
            append!(md.eqs_b, b[idxs])
            push!(md.eqs_A, A[idxs,:])
        elseif cone == :SOC
            function constr_fn(x)
                expr = b[idxs] - A[idxs, :]*x;
                return expr[end].^2 - sum(expr[1:end-1].^2);
            end
            push!(md.ineq_fns, constr_fn);
            push!(md.ineq_idxs, idxs);
        elseif cone == :SOCRotated
            function constr_fn(x)
                expr = b[idxs] - A[idxs, :]*x;
                return expr[end-1]*expr[end] - sum(expr[1:end-2].^2)
            end
            push!(md.ineq_fns, constr_fn);
            push!(md.ineq_idxs, idxs);
        elseif cone in [:SDP, :ExpPrimal, :ExpDual]
            throw(ArgumentError("Haven't coded feasibility for these cones yet."));
        elseif cone == :Free
            pass;
        else
            print("This cone is not recognized.");
        end
    end
    update_bounds!(md, l, u);
    @assert length(constr_cones) == length(md.ineqs_b) + length(md.eqs_b) + length(md.ineq_fns) +
                                    length(md.eq_fns)
    return md
end

# Solving a MOI model.
filename = "../../data/cblib.zib.de/flay03m.cbf.gz";
solver = Mosek.Optimizer();
mof_model = CBF_to_MOF(filename);
inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices());
MOI.optimize!(mof_model);
status = MOI.get(mof_model, MOI.TerminationStatus());
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
mof_vars = [MOI.get(mof_model, MOI.VariablePrimal(), var) for var in inner_variables];
@test mof_obj ≈ 48.989798037382


# # Getting fn_model using MPB
filename = "../../data/cblib.zib.de/flay03m.cbf.gz";
md = CBF_to_ModelData(filename);

# # Fitting function model (and updating bounds
# n_samples = 1000;
# n_dims = length(fn_model.lbs);
# fn_model.lbs = [maximum([fn_model.lbs[i], mof_vars[i]-40]) for i=1:n_dims]
# fn_model.ubs = [minimum([fn_model.ubs[i], mof_vars[i]+40]) for i=1:n_dims]
# example_fit(fn_model, lnr=base_otc())
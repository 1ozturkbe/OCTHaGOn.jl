using Test
using PyCall
using MathOptInterface, MosekTools
const MOI = MathOptInterface
using MathProgBase
using ConicBenchmarkUtilities

include("../src/structs.jl");

function alphac_to_fn(alpha, c; lse=false)
    nterms, xdim = size(alpha)
    if lse
        x -> sum([c[i]*exp(sum([alpha[i,j]*x[j] for j=1:xdim])) for i=1:nterms])
    else
        x -> sum([c[i]*prod([x[j]^alpha[i,j] for j=1:xdim]) for i=1:nterms])
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
    xdim = size(f.alpha,2);
    obj = alphac_to_fn(f.alpha, f.c; lse=lse);
    obj_idxs = unique([idx[2] for idx in findall(i->i != 0, f.alpha)]);
    ineqs = Vector{Function}();
    ineq_idxs = Array[];
    eqs = Vector{Function}();
    eq_idxs = Array[];
    lbs = Array{Union{Missing,Float64}}(missing, 1, xdim);
    ubs = Array{Union{Missing,Float64}}(missing, 1, xdim);
    for i=1:size(greaters,1)
        alpha = greaters[i].alpha
        c = greaters[i].c
        local idxs = findall(x->x!=0, alpha);
        if size(idxs, 1) > 1
            append!(ineqs, [alphac_to_fn(alpha, c; lse=lse)])
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
        alpha = equals[i].alpha;
        c = equals[i].c;
        idxs = findall(x->x!=0, alpha);
        append!(eqs, [alphac_to_fn(alpha, c; lse=lse)]);
        idxs = findall(x->x!=0, equals[i].alpha);
        append!(eq_idxs, [unique([idx[2] for idx in idxs])])
    end
    ex = function_model(string("example", idx),
                        obj, obj_idxs, ineqs, ineq_idxs,
                        eqs, eq_idxs, lbs, ubs, lse);
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
#     inner_variables = MOI.get(mof_model, MOI.ListOfVariableIndices())
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

function CBF_to_fnmodel(filename)
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
    num_vars = length(c);
    obj = x -> sum(c.*x);
    obj_idxs = findall(x -> x .!=0, c);
    # Variable bounds
    l = fill(-Inf, length(c));
    u = fill(Inf, length(c));
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
    ineqs = [];
    ineq_idxs = Array[];
    eqs = [];
    eq_idxs = Array[];
    for (cone, idxs) in constr_cones
    # Idxs describe which rows of A the cone applies to.
    # All cones: [:Free, :Zero, :NonNeg, :NonPos, :SOC, :SOCRotated, :SDP, :ExpPrimal, :ExpDual]
        if cone == :NonNeg
            if length(A[idxs,:].nzval) > 1.
                constr_fn = x -> b[idxs] - A[idxs, :]*x;
                push!(ineqs, constr_fn);
                push!(ineq_idxs, idxs);
            else
                l[findall(!iszero, A[idxs,:])[1][2]] = maximum([l[findall(!iszero, A[idxs,:])[1][2]],
                                                                b[idxs][1]/A[idxs, :].nzval[1]]);
            end
        elseif cone == :NonPos
            if length(idxs) > 1.
                constr_fn = x -> -1 .* b[idxs] + A[idxs, :]*x;
                push!(ineqs, constr_fn);
                push!(ineq_idxs, idxs);
            else
                u[findall(!iszero, A[idxs,:])[1][2]] = minimum([u[findall(!iszero, A[idxs,:])[1][2]],
                                                   b[idxs][1]/A[idxs, :].nzval[1]]);
            end
        elseif cone == :Zero
            constr_fn = x -> b[idxs] - A[idxs, :]*x;
            push!(eqs, constr_fn);
            push!(eq_idxs, idxs);
        elseif cone == :SOC
            function constr_fn(x)
                expr = b[idxs] - A[idxs, :]*x;
                return sum(expr[1:end-1].^2) <= expr[end].^2;
            end
            push!(ineqs, constr_fn);
            push!(ineq_idxs, idxs);
        elseif cone == :SOCRotated
            function constr_fn(x)
                expr = b[idxs] - A[idxs, :]*x;
                return sum(expr[1:end-2].^2) <= expr[end-1]*expr[end]
            end
            push!(ineqs, constr_fn);
            push!(ineq_idxs, idxs);
        elseif cone in [:SDP, :ExpPrimal, :ExpDual]
            print("Haven't coded feasibility for these cones yet.");
        elseif cone == :Free
            pass;
        else
            print("This cone is not recognized.");
        end
    end
    ex = function_model(filename,
                        obj, obj_idxs, ineqs, ineq_idxs,
                        eqs, eq_idxs, l, u, false);
    return ex
end

# Solving a MOI model.
filename = "../../data/cblib.zib.de/sambal.cbf.gz"
solver = Mosek.Optimizer();
mof_model = CBF_to_MOF(filename);
MOI.optimize!(mof_model);
status = MOI.get(mof_model, MOI.TerminationStatus());
mof_obj = MOI.get(mof_model, MOI.ObjectiveValue());
@test mof_obj ≈ 3.968224074

# Getting fn_model using MPB
filename = "../../data/cblib.zib.de/flay03m.cbf.gz";
fn_model = CBF_to_fnmodel(filename);


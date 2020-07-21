#=
tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Adds tools to mainly import optimization problems from other sources.
=#

using ConicBenchmarkUtilities
using MosekTools
using PyCall
using MathOptInterface

include("model_data.jl")
include("fit.jl")

function CBF_to_ModelData(filename; epsilon=1e-20)
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
    bad_cones = [:SDP];
    for (cone,idxs) in var_cones
        cone in bad_cones && error("Cone type $(cone) not supported")
    end
    n_vars = length(c);
    md = ModelData(c=c);
    # Variable bounds
    l = fill(-Inf, n_vars);
    u = fill(Inf, n_vars);
    for (cone,idxs) in var_cones
        if cone == :ExpPrimal
            l[idxs[2]] = 0;
        elseif cone == :ExpDual
            l[idxs[3]] = 0;
        elseif cone == :SOC
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
        var_idxs = unique(cart_ind[2] for cart_ind in findall(!iszero, A[idxs, :])); # CartesianIndices...
        if cone == :NonNeg
            if length(var_idxs) == 1.
                u[var_idxs[1]] = minimum([u[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            push!(md.ineqs_b, b[idxs]);
            push!(md.ineqs_A, A[idxs, :]);
        elseif cone == :NonPos
            if length(var_idxs) == 1.
                l[var_idxs[1]] = maximum([l[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            push!(md.ineqs_b, -b[idxs]);
            push!(md.ineqs_A, -1 .* A[idxs, :]);
        elseif cone == :Zero
            push!(md.eqs_b, b[idxs]);
            push!(md.eqs_A, A[idxs,:]);
        elseif cone == :SOC
            constr_fn = let b = b[idxs], A = A[idxs, :]
                function (x)
                    expr = b - A*x;
                    return expr[1].^2 - sum(expr[2:end].^2);
                end
            end
            add_ineq_fn!(md, constr_fn, var_idxs);
#             l[var_idxs[1]] = maximum([l[var_idxs[1]], 0]);
        elseif cone == :SOCRotated
            constr_fn = let b = b[idxs], A = A[idxs, :]
                function (x)
                    expr = b - A*x;
                    return expr[1]*expr[2] - sum(expr[3:end].^2)
                end
            end
            add_ineq_fn!(md, constr_fn, var_idxs);
#             l[var_idxs[1]] = maximum([l[var_idxs[1]], 0]);
#             l[var_idxs[2]] = maximum([l[var_idxs[2]], 0]);
        elseif cone == :ExpPrimal
            constr_fn = let b = b[idxs], A = A[idxs, :]
                function (x)
                    (x,y,z) = b - A*x;
                    return z - y*exp(x/y)
                end
            end
            add_ineq_fn!(md, constr_fn, var_idxs);
#             l[var_idxs[2]] = maximum([l[var_idxs[2]], epsilon]);
        elseif cone == :ExpDual
            constr_fn = let b = b[idxs], A = A[idxs, :]
                function (x)
                    (u,v,w) = b - A*x;
                    if v >= 0 && w >= 0
                        return 1 # feasible
                    elseif u < 0 && w >= 0
                        return u*log(-u/w) - u + v
                    else
                        return -1 # infeasible
                    end
                end
            end
            add_ineq_fn!(md, constr_fn, var_idxs);
#             u[var_idxs[1]] = minimum([u[var_idxs[1]], 0]);
#             l[var_idxs[3]] = maximum([l[var_idxs[3]], 0]);
        elseif cone in [:SDP]
            throw(ArgumentError("Haven't coded feasibility for these cones yet."));
        elseif cone == :Free
            pass;
        else
            print("This cone is not recognized.");
        end
    end
    update_bounds!(md, l, u);
    md.int_idxs = findall(x -> x .== :Int, vartypes); # Integer var labeling.
    return md
end

function alphac_to_fn(alpha, c; lse=false)
    n_terms, n_vars = size(alpha)
    if lse
        x -> sum([c[i]*exp(sum([alpha[i,j]*x[j] for j=1:n_vars])) for i=1:n_terms])
    else
        x -> sum([c[i]*prod([x[j]^alpha[i,j] for j=1:n_vars]) for i=1:n_terms])
    end
end

function sagemark_to_ModelData(idx; lse=false)
    """
    Imports sagebenchmarks example from literature.solved and
    returns as a function_model.
    Arguments:
        idx:: number of benchmark
    Returns:
        md:: a function_model
    """
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(idx);
    f, greaters, equals = signomials;
    n_vars = size(f.alpha,2)+1;
    # Turning objective into constraint
    c = zeros(n_vars); c[end] = 1;
    md = ModelData(name = string("sagemark", idx), c = c);
    obj_c = vcat(-1 .* f.c, [1]);
    obj_alpha = vcat(-1 .* f.alpha, zeros(size(f.alpha,2))');
    obj_alpha = hcat(obj_alpha, zeros(length(obj_c)));
    obj_alpha[end, end] = 1;
    obj_constr = alphac_to_fn(obj_alpha, obj_c, lse=lse)
    idxs = unique([idx[2] for idx in findall(i->i != 0, obj_alpha)]);
    add_ineq_fn!(md, obj_constr, idxs);
    # Bound initialization
    lbs = -Inf*ones(n_vars);
    ubs = Inf*ones(n_vars);
    for i=1:length(greaters)
        alpha = hcat(greaters[i].alpha, zeros(size(greaters[i].alpha,1)));
        c = greaters[i].c;
        local idxs = findall(x->x!=0, alpha);
        if size(idxs, 1) > 1
            add_ineq_fn!(md, alphac_to_fn(alpha, c; lse=lse), unique([idx[2] for idx in idxs]))
            if sum(float(c .>= zeros(length(c)))) == 1
                if lse
                    push!(md.convex_idxs, length(md.ineq_fns))
                else
                    push!(md.logconvex_idxs, length(md.ineq_fns))
                end
            end
        else
            val = -((sum(c)-c[idxs[1][1]]) / c[idxs[1][1]])^(1/alpha[idxs[1]]);
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
    for i=1:length(equals)
        alpha = hcat(equals[i].alpha, zeros(size(equals[i].alpha,1)));
        idxs = findall(x->x!=0, alpha);
        add_eq_fn!(md, alphac_to_fn(alpha, c; lse=lse), unique([idx[2] for idx in idxs]));
    end
    update_bounds!(md, lbs, ubs);
    return md
end

function CBF_to_MOF(filename; solver=Gurobi.Optimizer())
    """ Imports a conic benchmark into a MathOptInterface format. """
    model = MathOptInterface.FileFormats.Model(filename = filename);
    mof_model = MathOptInterface.Bridges.full_bridge_optimizer(solver , Float64)
    MathOptInterface.read_from_file(model, filename)
    MathOptInterface.copy_to(mof_model, model)
    return mof_model
end

# function MOF_to_ModelData(mof_model)
# TODO: Work in progress...
#     constraints = MathOptInterface.get(mof_model, MathOptInterface.ListOfConstraints())
#     constraint_indices = MathOptInterface.get(mof_model, MathOptInterface.ListOfConstraintIndices)
#     variable_attributes_set = MathOptInterface.get(mof_model, MathOptInterface.ListOfVariableAttributesSet())
#     #TODO: CONTINUE HERE
#     obj = model.objective.constant*model.objective.terms
#     lbs = model.lower_bounds;
#     ubs = model.upper_bounds;
#     ex = function_model(string("example", idx),
#                         obj, obj_idxs, ineqs, ineq_idxs,
#                         eqs, eq_idxs, lbs, ubs, lse);
#     return ex
# end
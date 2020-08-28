#=
tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Adds tools to mainly import optimization problems from other sources.
=#

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
        vks = [Symbol("x", i) for i in var_idxs];
        if cone == :NonNeg
            if length(var_idxs) == 1.
                u[var_idxs[1]] = minimum([u[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            add_linear_ineq!(md, A[idxs,:], b[idxs]);
        elseif cone == :NonPos
            if length(var_idxs) == 1.
                l[var_idxs[1]] = maximum([l[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            add_linear_ineq!(md, -1. .* A[idxs,:], -b[idxs]);
        elseif cone == :Zero
            add_linear_eq!(md, A[idxs,:], b[idxs]);
        elseif cone == :SOC
            constr_fn = let b = b[idxs], A = A[idxs, var_idxs], vks = vks
                function (x)
                    vars = [x[vk] for vk in vks];
                    expr = b - A*vars;
                    return expr[1].^2 - sum(expr[2:end].^2);
                end
            end
            add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = vks));
            l[var_idxs[1]] = maximum([l[var_idxs[1]], 0]);
        elseif cone == :SOCRotated
            constr_fn = let b = b[idxs], A = A[idxs, var_idxs], vks = vks
                function (x)
                    vars = [x[vk] for vk in vks];
                    expr = b - A*vars;
                    return expr[1]*expr[2] - sum(expr[3:end].^2)
                end
            end
            add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = vks));
            l[var_idxs[1]] = maximum([l[var_idxs[1]], 0]);
            l[var_idxs[2]] = maximum([l[var_idxs[2]], 0]);
        elseif cone == :ExpPrimal
            constr_fn = let b = b[idxs], A = A[idxs, var_idxs], vks = vks
                function (x)
                    vars = [x[vk] for vk in vks];
                    (x,y,z) = b - A*vars;
                    return z - y*exp(x/y)
                end
            end
            add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = vks));
            l[var_idxs[2]] = maximum([l[var_idxs[2]], epsilon]);
        elseif cone == :ExpDual
            constr_fn = let b = b[idxs], A = A[idxs, var_idxs], vks = vks
                function (x)
                    vars = [x[vk] for vk in vks];
                    (u,v,w) = b - A*vars;
                    if v >= 0 && w >= 0
                        return 1 # feasible
                    elseif u < 0 && w >= 0
                        return u*log(-u/w) - u + v
                    else
                        return -1 # infeasible
                    end
                end
            end
            add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = vks));
            u[var_idxs[1]] = minimum([u[var_idxs[1]], 0]);
            l[var_idxs[3]] = maximum([l[var_idxs[3]], 0]);
        elseif cone in [:SDP]
            throw(ArgumentError("Haven't coded feasibility for these cones yet."));
        elseif cone == :Free
            pass;
        else
            print("This cone is not recognized.");
        end
    end
    update_bounds!(md, lbs = Dict(md.vks .=> l),
                       ubs = Dict(md.vks .=> u));
    md.int_vks = [Symbol("x", i) for i in findall(x -> x .== :Int, vartypes)]; # Integer var labels
    return md
end

function alphac_to_fn(alpha, c; lse=false)
    n_terms, n_vars = size(alpha)
    idxs = unique([i[2] for i in findall(i->i != 0, alpha)]);
    alpha = alpha[:,idxs];
    vks = [Symbol("x", i) for i in idxs];
    if lse
        x -> sum([c[i]*exp(sum([alpha[i,j]*x[vks[j]] for j=1:length(vks)])) for i=1:n_terms])
    else
        x -> sum([c[i]*prod([x[vks[j]]^alpha[i,j] for j=1:length(vks)]) for i=1:n_terms])
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
    obj_alpha = vcat(f.alpha, zeros(size(f.alpha,2))');
    obj_alpha = hcat(obj_alpha, zeros(length(obj_c)));
    obj_alpha[end, end] = 1;
    obj_fn = alphac_to_fn(obj_alpha, obj_c, lse=lse);
    add_fn!(md, BlackBoxFunction(fn = obj_fn, vks = obj_fn.vks));
    # Bound initialization
    for i=1:length(greaters)
        alpha = hcat(greaters[i].alpha, zeros(size(greaters[i].alpha,1)));
        c = greaters[i].c;
        constr_fn = alphac_to_fn(alpha, c, lse=lse);
        if length(constr_fn.vks) > 1
            add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = constr_fn.vks));
            if sum(float(c .>= zeros(length(c)))) == 1
                if lse
                    push!(md.fns[end].tags, "convex")
                else
                    push!(md.fns[end].tags, "logconvex")
                end
            end
        else
            idx = unique([i for i in findall(i->i != 0, alpha)])[1]; #idx[1] is monomial index,
            vk = md.vks[idx[2]];                                     #idx[2] is variable index.
            val = -((sum(c)-c[idx[1]]) / c[idx[1]])^(1/alpha[idx]);
            if lse
                val=log(val)
            end
            if c[idx[1]] <= 0
                update_bounds!(md, ubs = Dict(md.vks[idx[2]] => val));
            else
                update_bounds!(md, lbs = Dict(md.vks[idx[2]] => val));
            end
        end
    end
    for i=1:length(equals)
        alpha = hcat(equals[i].alpha, zeros(size(equals[i].alpha,1)));
        c = equals[i].c;
        constr_fn = alphac_to_fn(alpha, c, lse=lse);
        add_fn!(md, BlackBoxFunction(fn = constr_fn, vks = constr_fn.vks, equality = true));
    end
    return md
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
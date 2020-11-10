#=
cbf:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-10-08
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
        vks = Dict(:x => [var_idxs])
        if cone == :NonNeg
            if length(var_idxs) == 1.
                u[var_idxs[1]] = minimum([u[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            add_lin_constr!(md, @build_constraint(A[idxs,:] * [md.vars[vk] for vk in md.vks] .<= b[idxs]));
        elseif cone == :NonPos
            if length(var_idxs) == 1.
                l[var_idxs[1]] = maximum([l[var_idxs[1]], b[idxs][1]/A[idxs, :].nzval[1]]);
            end
            add_lin_constr!(md, @build_constraint(A[idxs,:] * [md.vars[vk] for vk in md.vks] .>= b[idxs]));
        elseif cone == :Zero
            add_lin_constr!(md, @build_constraint(A[idxs,:] * [md.vars[vk] for vk in md.vks] .== b[idxs]));
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
    int_vks = [Symbol("x", i) for i in findall(x -> x .== :Int, vartypes)]; # Integer var labels
    for vk in int_vks
        set_integer(md.vars[vk])
    end
    return md
end
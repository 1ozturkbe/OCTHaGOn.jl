using Test
using PyCall
using MathOptInterface, MosekTools
const MOI = MathOptInterface

include("../src/structs.jl")

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
    user_model = MOI.Bridges.full_bridge_optimizer(solver , Float64)

    MOI.read_from_file(model, filename)
    MOI.copy_to(user_model, model)
    return user_model
end

filename = "../../data/cblib.zib.de/flay02m.cbf.gz";
solver = Mosek.Optimizer();
model = CBF_to_MOF(filename);
MOI.optimize!(model)
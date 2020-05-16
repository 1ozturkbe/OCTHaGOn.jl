using Test
include("../src/structs.jl")
using PyCall

function alphac_to_fn(alpha, c)
    nterms, xdim = size(alpha)
    sig_fn(x) = sum([c[i]*exp(sum(alpha[i,j]*x[j] for j=1:xdim)) for i=1:nterms])
    return sig_fn
end

function import_sagebenchmark(exidx)
    """
    Imports sagebenchmarks example from literature.solved and
    returns as a function_model.
    """
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(exidx);
    f, gts, eqs = signomials;
    xdim = size(f.alpha,2);
    obj(x) = alphac_to_fn(f.alpha, f.c);
    obj_idxs = unique([idx[2] for idx in findall(i->i != 0, f.alpha)]);
    constr = Vector{Function}();
    constr_idxs = Array[];
    lbs = Array{Union{Missing,Float64}}(missing, 1, xdim);
    ubs = Array{Union{Missing,Float64}}(missing, 1, xdim);
    for i=1:size(gts,1)
        alpha = gts[i].alpha;
        c = gts[i].c;
        idxs = findall(x->x!=0, alpha);
        if size(idxs, 1) > 1
            append!(constr, [alphac_to_fn(alpha, c)]);
            append!(constr_idxs, [unique([idx[2] for idx in idxs])]);
        else
            local c_rat = -c[idxs[1][1]] / (sum(c)-c[idxs[1][1]]);
            local val = 1/alpha[idxs[1]]*log(c_rat);
            local coeff = alpha[idxs[1]]*c[idxs[1][1]];
            if coeff >= 0 && (ismissing(ubs[idxs[1][2]]) || ubs[idxs[1][2]] >= val)
                ubs[idxs[1][2]] = val;
            elseif ismissing(lbs[idxs[1][2]]) || lbs[idxs[1][2]] <= val
                lbs[idxs[1][2]] = val;
            end
        end
    end
    ex = function_model(string("example", exidx),
                        obj, obj_idxs, constr, constr_idxs, lbs, ubs);
    return ex
end


function example1()
    obj(x) = 0.5*exp(x[1]-x[2]) - exp(x[1]) - 5*exp(-x[2]);
    obj_idxs = [1,2]
    g1(x) = 100 - exp(x[2]-x[3]) - exp(x[2]) - 0.05*exp(x[1]+x[3]);
    constr = [g1]
    constr_idxs = [[1,2,3]]
    lbs = log.([70, 1, 0.5]);
    ubs = log.([150, 20, 21]);
    ex = function_model("example1", obj, obj_idxs, constr, constr_idxs, lbs, ubs)
    return ex
end

function example2()
    obj(x) = x[1]
    obj_idxs = [1]
    g1(x) = 1 - 3.7/x[1]*x[2]^0.85 - 1.98/x[1]*x[2] - 700.3/x[1]*x[3]^-0.75
    g2(x) = 1 - 0.7673*x[3]^0.05 + 0.05* x[2]
    constr = [g1, g2]
    constr_idxs = [[1,2,3], [2,3]]
    lbs = [5, 0., 380]
    ubs = [15, 5, 450]
    ex = function_model("example2", obj, obj_idxs, constr, constr_idxs, lbs, ubs)
end

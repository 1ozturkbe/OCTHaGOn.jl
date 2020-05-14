using Test
include("../structs.jl")
using PyCall

function import_sagebenchmark(number)
    u
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


using LatinHypercubeSampling
include("../src/fit.jl");
include("../src/constraintify.jl")
include("examples.jl");

function compile_lse_constraints()
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    ineq_constraints = Function[];
    convex = Bool[];
    idxs = Int64[];
    for idx=1:25
        fn_model = import_sagebenchmark(idx, lse = true);
        if !any(ismissing.(fn_model.lbs)) && !any(ismissing.(fn_model.ubs))
            println("Adding constraints from test ", string(idx))
            signomials, solver, run_fn = sagemarks.get_example(idx);
            f, greaters, equals = signomials;
            xdim = size(f.alpha,2);
            # Only testing inequality constraints for now...
            for j in greaters
                if abs(sum(j.alpha)) != 1.
                    append!(idxs, [idx])
                    append!(ineq_constraints, [alphac_to_fn(j.alpha, j.c, lse = true)])
                    if sum(float(j.c .>= 0.)) == 1
                        append!(convex, [true])
                    else
                        append!(convex, [false])
                    end
                end
            end
        end
    end
    println("Compiled all multivariate LSE inequalities!")
    return ineq_constraints, convex, idxs
end

ineqs, convex, idxs = compile_lse_constraints()
n_samples = 1000;
for i=1:length(ineqs)
    print("Fitting constraint ", i, ".")
    fn_model = import_sagebenchmark(idxs[i], lse=true)
    n_dims = length(fn_model.lbs);
    plan, _ = LHCoptim(n_samples, n_dims, 1);
    X = scaleLHC(plan,[(fn_model.lbs[i], fn_model.ubs[i]) for i=1:n_dims]);
    feas_tree = learn_constraints(base_otc(), [ineqs[i]], X)
    IAI.write_json(string("data/constraint",i,".tree"), feas_tree[1])
end
lnrs = [IAI.read_json(string("data/constraint",i,".tree")) for i=1:length(ineqs)]

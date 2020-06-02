include("../src/fit.jl");
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

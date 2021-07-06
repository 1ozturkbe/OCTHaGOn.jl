# Tests convexity detection functions. 
function gen_quad_expr(n::Int64 = 10)
    A = -5 .+ 10 .* rand(n, n)
    b = -5 .+ 10 .* rand(n)
    c = -5 + 10*rand()
    expr = :(x -> x .* $(A) .* x + sum(b .* x) + c)
    return expr
end

idxs = 1:25; # 25
# max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
gms = Dict()
for idx in idxs
    try
        ex = sagemark_to_GlobalModel(idx, lse=true);
        gms[idx] = ex
    catch
        println("Failed import of problem $(idx).")
    end
end

# Finding the bounded BBLs
bounded_bbls = []
for idx in idxs
    gm = gms[idx]
    find_bounds!(gm)
    for bbl in gm.bbls
        try
            check_bounds(get_bounds(bbl))
            bbl.name = gm.name * "_" * bbl.name
            push!(bounded_bbls, bbl)
        catch ErrorException
        end
    end
end
@info "Found $(length(bounded_bbls)) bounded BBLs."

# Making sure to compute the convexity of the BBLs. 
for bbl in bounded_bbls
    uniform_sample_and_eval!(bbl)
    update_vexity(bbl)
end
@info "Found $(sum(bbl.convex for bbl in bounded_bbls)) convex BBLs."

# Actually confirming convexity of the underlying functions. 
convex_ones = Dict(1 => [1,3], 2 => [1], 19 => [1], 21 => [4], 23 => [4])

for (key, values) in convex_ones
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(key);
    f, greaters, equals = signomials;

    for i = 1:length(greaters)
        constrexpr, constrvars = alphac_to_expr(gm.model, greaters[i].alpha, greaters[i].c, lse=lse)
    end
    for i = 1:length(equals)
        constrexpr, constrvars = alphac_to_expr(gm.model, equals[i].alpha, equals[i].c, lse=lse)
    end
end

# filenums = [2.15, 2.16, 2.17, 2.18, 3.13]
# filenames = ["problem" * string(filenum) * ".gms" for filenum in filenums]
# gms = Dict()
# for filename in filenames
#     try
#         gm = GAMS_to_GlobalModel(OCT.GAMS_DIR, filename)
#         types = JuMP.list_of_constraint_types(gm.model)
#         if !isempty(types)
#             total_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
#         end
#         gms[filename] = gm
#     catch
#         throw(OCTException(filename * " has an import issue."))
#     end
# end

# gm = gms[filenames[1]]
# set_optimizer(gm, CPLEX.Optimizer)
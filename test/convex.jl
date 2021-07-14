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
        ex = sagemark_to_GlobalModel(idx, true);
        gms[idx] = ex
    catch
        println("Failed import of problem $(idx).")
    end
end

# Finding the bounded BBLs
bounded_bbls = []
bbl_idxs = Dict(idx => [] for idx in collect(idxs))
for idx in idxs
    gm = gms[idx]
    find_bounds!(gm)
    ct = 0
    for i = 1:length(gm.bbls)
        ct += 1
        try
            check_bounds(get_bounds(gm.bbls[i]))
            gm.bbls[i].name = gm.name * "_" * gm.bbls[i].name
            push!(bounded_bbls, gm.bbls[i])
            push!(bbl_idxs[idx], ct)
        catch ErrorException
        end
    end
end
bbl_idxs = Dict(key => val for (key, val) in bbl_idxs if !isempty(val))

@info "Found $(length(bounded_bbls)) bounded BBLs."

# Making sure to compute the convexity of the BBLs. 
for bbl in bounded_bbls
    uniform_sample_and_eval!(bbl)
    update_vexity(bbl)
end
@info "Found $(sum(bbl.convex for bbl in bounded_bbls)) convex BBLs."
convex_bbls = [bbl for bbl in bounded_bbls if bbl.convex]

# Actually confirming convexity of the underlying functions. 
convex_ones = Dict(1 => [1,3], 2 => [1], 19 => [1], 21 => [4], 23 => [4])

""" Checks the convexity of signomial constraints (sig >= 0). """
function check_cvx_con(alpha, c, lse::Bool = true)
    cvxity = true
    if lse # exponential form
        if length(c) == 1
        elseif length(c) >= 2
            if sum(c .> 0) != 1
                cvxity = false
            end
        end
    else # geometric form
        for i in 1:length(c)
            if c[i] >= 0
                if any(alpha[i,:] .> 0)
                    cvxity = false
                    break
                end
            else
                if any(alpha[i,:] .< 0)
                    cvxity = false
                    break
                end
            end
        end
    end
    return cvxity
end

""" Checks the convexity of signomial objectives. """
function check_cvx_obj(alpha, c, lse::Bool = true)
    cvxity = true
    if lse # exponential form
        if length(c) == 1
        elseif length(c) >= 2
            if any(c .< 0) 
                cvxity = false
            end
        end
    else # geometric form
        for i in 1:length(c)
            if c[i] >= 0
                if any(alpha[i,:] .< 0)
                    cvxity = false
                    break
                end
            else
                if any(alpha[i,:] .> 0)
                    cvxity = false
                    break
                end
            end
        end
    end
    return cvxity
end

# Finding the actual convex ones analytically
lse = true
actual_cvx = Dict(idx => [] for idx in collect(idxs))

for (idx, val) in bbl_idxs
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(idx);
    f, greaters, equals = signomials;
    ct = 1
    check_cvx_obj(f.alpha, f.c, lse) && push!(actual_cvx[idx], ct)
    for i = 1:length(greaters)
        # if to prune the bounds
        if !(sum(greaters[i].alpha .!= 0) == 1 && sum(greaters[i].alpha) == 1) && ct in bbl_idxs[idx]
            ct += 1
            check_cvx_con(greaters[i].alpha, -1 .* greaters[i].c, lse) && 
            push!(actual_cvx[idx], ct)
        end
    end
    for i = 1:length(equals)
        if !(sum(equals[i].alpha .!= 0) == 1 && sum(equals[i].alpha) == 1) && ct in bbl_idxs[idx]
            ct += 1
            check_cvx_con(equals[i].alpha, -1 .* equals[i].c, lse) && 
            push!(actual_cvx[idx], ct)
        end    
    end
end
actual_cvx = Dict(key => value for (key, value) in actual_cvx if !isempty(value))

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

# Functions to check
# g(x) = x^2/(x^2+1)


# g(x) = x[1]^4*x[2]^2 + x[2]^4*x[1]^2 - 3x[1]^2*x[2]^2 + 1 # Motzkin polynomial
m = Model()
@variable(m, -2 <= x[1:2] <= 2)
gm = GlobalModel(model = m)
add_nonlinear_constraint(gm, :(x -> x[1]^4*x[2]^2 + x[2]^4*x[1]^2 - 3x[1]^2*x[2]^2 + 1))
uniform_sample_and_eval!(gm)
update_vexity.(gm.bbls)
@test !gm.bbls[1].convex

# An example from : https://arxiv.org/pdf/0903.1287v1.pdf
m = Model()
@variable(m, -1 <= x[1:3] <= 1)
gm = GlobalModel(model = m)
add_nonlinear_constraint(gm, :(x -> 32*x[1]^8 + 118*x[1]^6*x[2]^2 + 40*x[1]^6*x[3]^2 + 25*x[1]^4*x[2]^4 -
                            43*x[1]^4*x[2]^2*x[3]^2 - 35*x[1]^4*x[3]^4 + 3*x[1]^2*x[2]^4*x[3]^2 - 
                            16*x[1]^2*x[2]^2*x[3]^4 + 24*x[1]^2*x[3]^6 + 16*x[2]^8 + 
                            44*x[2]^6*x[3]^2 + 70*x[2]^4*x[3]^4 + 60*x[2]^2*x[3]^6 + 30*x[3]^8))
update_vexity(gm.bbls[1])
@test gm.bbls[1].convex
# 

# The implications of convexity finding. 
# This allows us to exploit the properties of convex functions in the domains they are relevant, and come up with appropriate heuristic in the domains where they are invalid. 
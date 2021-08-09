function test_baron_solve(m::JuMP.Model = gear(false))
    set_optimizer(m, BARON_SILENT)
    optimize!(m)
    sol = solution(m)
    @test true
end

function test_speed_params(gm::GlobalModel = minlp(true), solver = CPLEX_SILENT)
    set_optimizer(gm, solver)   
    bbl = gm.bbls[1]
    uniform_sample_and_eval!(bbl)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [3, 5]
    tree_mat = [[], []]
    time_mat = [[], []]
    for i=1:length(ls_num_hyper_restarts)
        for j=1:length(ls_num_tree_restarts)
            t1 = time()
            params = Dict(:ls_num_hyper_restarts => ls_num_hyper_restarts[i],
                          :ls_num_tree_restarts => ls_num_tree_restarts[j],
                          :max_depth => 2)
            learn_constraint!(bbl; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbl.learners[end])
        end
    end
    @test true
end

function test_classify_gradients()
    gm = minlp(true)
    bbr = gm.bbls[3]
    uniform_sample_and_eval!(gm)
    idxs = collect(1:10)
    classify_curvature(bbr, idxs)
    @test !any(ismissing.(bbr.curvatures[idxs]))
    classify_curvature(bbr)
    @test all(bbr.curvatures .> 0)
    update_vexity(bbr)
    @test bbr.convex == true
    @test bbr.local_convexity == 1.
end

function test_infeasibility_cuts()
    gm = sagemark_to_GlobalModel(15, false)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    add_tree_constraints!(gm)
    optimize!(gm)
    bbc_idxs = [bbl isa BlackBoxClassifier for bbl in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_feasibility_sample()
    gm = speed_reducer()
    uniform_sample_and_eval!(gm)
    [set_param(bbl, :threshold_feasibility, 0.3) for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    @test any(check_feasibility(gm) .!= 1)
    feasibility_sample(gm)
    @test all(check_feasibility(gm) .== 1)
end

function test_survey_method(gm::GlobalModel = minlp(true))
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    surveysolve(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    bbc_idxs = [x isa BlackBoxClassifier for x in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_concave_regressors(gm::GlobalModel = gear(true))
    gm = gear(true)
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        update_vexity.(bbrs)  
    end
    bbr = bbrs[1]
    
    # Checking number of constraints
    types = JuMP.list_of_constraint_types(gm.model)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    surveysolve(gm) # 1st tree (ORT)
    actual = bbr.actuals[end]
    optim = bbr.optima[end]
    learn_constraint!(bbr, "upper" => minimum(bbr.actuals)) # 2nd tree (Upper OCT)
    learn_constraint!(bbr, "reg" => 40, 
                        regression_sparsity = 0, max_depth = 3) # 3th tree (Regressor on upper bounded samples)

    # Trying to add and remove individual constraints in random order to make sure no constraints accidentally remain. 
    for i=1:length(bbr.learners)
        update_tree_constraints!(gm, bbr, i)
        for j=1:length(bbr.learners)
            update_tree_constraints!(gm, bbr, j)
            for k = 1:length(bbr.learners)
                update_tree_constraints!(gm, bbr, k)
                treekeys = collect(keys(bbr.active_trees))
                treevalues = collect(values(bbr.active_trees))
                if length(treevalues) >= 2
                    @test Pair(treevalues[1].first, treevalues[2].first) in OCT.valid_pairs
                elseif length(treevalues) == 1
                    @test treevalues[1].first in OCT.valid_singles || treevalues[1].first == "upper"
                end
                clear_tree_constraints!(gm)
                n_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
                @test n_constraints == init_constraints
            end
        end
    end
    update_tree_constraints!(gm, bbr, 2)
    update_tree_constraints!(gm, bbr, 3)
    @test active_lower_tree(bbr) == 3
    @test active_upper_tree(bbr) == 3  
    optimize!(gm)
    clear_tree_constraints!(gm)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
end

function recipe(gm::GlobalModel)
    @info "GlobalModel " * gm.name * " in progress..."
    set_optimizer(gm, CPLEX_SILENT)
    uniform_sample_and_eval!(gm)
    set_param(gm, :ignore_accuracy, true)
    set_param(gm, :ignore_feasibility, true)
    for bbl in gm.bbls
        learn_constraint!(bbl)
        add_tree_constraints!(gm, bbl)
    end
    optimize!(gm)    
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while (gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
end 

function optimize_and_time!(m::Union{JuMP.Model, GlobalModel})
    t1 = time()
    if m isa JuMP.Model
        optimize!(m);
    else
        recipe(m)
    end
    println("Model solution time: " * string(time()-t1) * " seconds.")
end

# function refine_equality_tree()
#     gm = nlp3(true)
#     recipe(gm)
#     feas_tol = get_param(gm, :tighttol)
#     for bbl in gm.bbls
#         if bbl.equality && abs(bbl.feas_gap[end]) >= feas_tol
#             sol = DataFrame(gm.solution_history[end, string.(bbl.vars)])
#             learn_constraint!(bbl, sample_weight = reweight(bbl, sol))
#             update_tree_constraints!(gm, bbl)
#         end
#     end
#     if bbr.feas_gap[end] <= 0 && bbr.feas_gap[end]/max_gap <= -feas_tol 
#         update_tree_constraints!(gm, bbr)
#     end
#     @test true
# end

# test_baron_solve()

# test_speed_params()

# test_classify_gradients()

# test_infeasibility_cuts()

# test_feasibility_sample()

# test_survey_method()

# test_concave_regressors()

# Implementing gradient descent
m = JuMP.Model()
@variable(m, -1 <= x <= 4)
@variable(m, -1 <= y <= 4)
@variable(m, obj)
@objective(m, Min, obj)
gm = GlobalModel(model = m)
set_param(gm, :ignore_accuracy, true)
set_param(gm, :ignore_feasibility, true)

# add_nonlinear_constraint(gm, :(x -> 4*x[1]^3 + x[2]^2 + 2*x[1]^2*x[2]), dependent_var = obj)
# add_nonlinear_constraint(gm, :(x -> 20 - 3*x[1]^2 - 5*x[2]^2))

add_nonlinear_constraint(gm, :((x,y) -> 2*x^6 - 12.2*x^5 + 21.2*x^4 + 6.2*x - 6.4*x^3 - 4.7*x^2 + 
 y^6 - 11*y^5 + 43.3*y^4 - 10*y - 74.8*y^3 + 56.9*y^2 - 4.1*x*y - 0.1*y^2*x^2 + 0.4*y^2*x + 0.4*x^2*y), 
 name = "objective", dependent_var = obj)
add_nonlinear_constraint(gm, :((x,y) -> 10.125 - (x-1.5)^4 - (y-1.5)^4), name = "h1")
add_nonlinear_constraint(gm, :((x,y) -> (2.5 - x)^3 + (y+1.5)^3 - 15.75), name = "h2")

uniform_sample_and_eval!(gm)
learn_constraint!(gm)
add_tree_constraints!(gm)
set_optimizer(gm, CPLEX_SILENT)
optimize!(gm)

# Formulating descent algorithm
clear_tree_constraints!(gm)

x0 = gm.solution_history[end, :]

# Objective computations
obj_bbl = gm("objective")
gm_bounds = get_bounds(gm.bbls)
gm_vars = all_variables(gm.bbls)
sol_vals = Array(x0[string.(gm_vars)])
obj_gradient = DataFrame([Float64 for i=1:length(gm_vars)], string.(gm_vars))
append!(obj_gradient, DataFrame(string.(gm_vars) .=> obj_bbl.g(Array(x0[string.(obj_bbl.vars)]))),
        cols = :subset)
obj_gradient = coalesce.(obj_gradient, 0)
obj_value = obj_bbl.f(Array(x0[string.(obj_bbl.vars)])...)

# Descent direction
d = @variable(m, d[1:length(gm_vars)])
constrs = [@constraint(gm.model, sum(d .* Array(obj_gradient)) <= -1e-5), 
           @constraint(gm.model, d .== gm_vars - sol_vals)] # unit vector
@objective(gm.model, Min, sum(d .* Array(obj_gradient)))

# Implementing a binary localsearch algorithm from the last start point 

bbls = [bbl for bbl in gm.bbls if bbl != obj_bbl]
constr_grads = DataFrame([Float64 for i=1:length(gm_vars)], string.(gm_vars))
constr_vals = []
bbl_constrs = []
for bbl in bbls
    var_vals = Array(x0[string.(bbl.vars)])
    push!(constr_vals, bbl.f(var_vals...))
    new_grad = DataFrame(string.(bbl.vars) .=> bbl.g(var_vals))
    append!(constr_grads, new_grad, cols=:subset)
end
constr_grads = coalesce.(constr_grads, 0)

# Make sure to restore feasibility in case of infeasibility,
# and create a trust region 
for i = 1:length(constr_vals)
    push!(constrs, @constraint(gm.model, sum(Array(constr_grads[i,:]) .* d)  + constr_vals[i] >= 0))
end

# Finding the direction
optimize!(gm.model)
d_vals = getvalue.(d)

# Implementing linesearch (i.e. finding a good step size t)
var_bounds = get_bounds(gm.vars)
max_ts = 

# Plotting results

using Plots
colors = ["yellow", "orange", "red"]
plt = plot()
plt = quiver!([gm.solution_history[end, "x"]], [gm.solution_history[end,"y"]], 
    quiver = (obj_gradient[:, "x"]./sqrt.((obj_gradient[:, "x"].^2 .+ obj_gradient[:, "y"].^2)), 
                obj_gradient[:, "y"]./sqrt.((obj_gradient[:, "x"].^2 .+ obj_gradient[:, "y"].^2))), 
    markershape = :star5, color = "green", label = "+ Objective")
for i=1:length(bbls)
    bbl = bbls[i]
    infeas_idxs = findall(x -> x .< 0, bbl.Y)
    plt = scatter!(bbl.X[infeas_idxs,"x"], bbl.X[infeas_idxs, "y"], color = colors[i])
    plt = quiver!([gm.solution_history[end, "x"]], [gm.solution_history[end,"y"]], 
    quiver = (constr_grads[[i], "x"]./sqrt.((constr_grads[[i], "x"].^2 .+ constr_grads[[i], "y"].^2)), 
                constr_grads[[i], "y"]./sqrt.((constr_grads[[i], "x"].^2 .+ constr_grads[[i], "y"].^2))),
    markershape = :circle, color = colors[i], label = "+ BBL$(i)")
end
plt = quiver!([gm.solution_history[end, "x"]], [gm.solution_history[end,"y"]], 
    quiver = ([d_vals[1]], [d_vals[2]]),
    markershape = :square, color = "purple", label = "Descent direction")
display(plt)
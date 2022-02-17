
include("../load.jl")

using LinearAlgebra, Random, BARON, Pkg

algs = ["SVM", "CART", "OCT"]

function create_negative_definite_matrix(N)
    # Create a random matrix
    A = rand(N, N)

    # Create an orthogonal matrix
    U = eigvecs(A*A')

    # Create a negative-definite matrix Q
    Q = U * (Diagonal(rand(N)) .- 0.5) * U'
    
    return Q
end

"""
Creates a non-convex QP model with different numbers of variables/constraints.
Returns  a GlobalModel and a JuMP baron model 
"""
function create_noncvx_qp_models(n_vars, n_constr, linear_objective=true; seed=123)
    
    Random.seed!(seed)
    
    N = n_vars
    M = n_constr

    
    # Create upper and bounds (those particular values are just an experiment)
    lbs = push!(-5*ones(N), -800)
    ubs = push!(5*ones(N), 0)
    
    local function create_gm_model(obj_expr, constr_expr)
        
        # Initialize JuMP model
        m = JuMP.Model(with_optimizer(CPLEX_SILENT))
        @variable(m, x[1:N+1])
        @objective(m, Min, x[N+1])
        
        # Initialize GlobalModel
        gm = OCTHaGOn.GlobalModel(model = m, name = "qp")
        
        # Set bounds
        lbs_dict = Dict(x .=> lbs)
        ubs_dict = Dict(x .=> ubs)
        OCTHaGOn.bound!(gm, Dict(var => [lbs_dict[var], ubs_dict[var]] for var in gm.vars))
        
        # Add objective
        # @TODO: see why expr_vars argument is needed
        OCTHaGOn.add_nonlinear_or_compatible(gm, 
            obj_expr, vars=x[1:N], 
            expr_vars=[x[1:N]], dependent_var = x[N+1],
            is_objective=true
        )
        
        # Add constraints
        for constr in constr_expr
            OCTHaGOn.add_nonlinear_constraint(gm, constr, vars=x[1:N], expr_vars=[x[1:N]], alg_list=algs)
        end
        
        OCTHaGOn.set_param(gm, :ignore_accuracy, true)
        #OCTHaGOn.set_param(gm, :ignore_feasibility, true)
        
        return gm
    end
    
    local function create_baron_model(obj_expr, constr_expr)
        
        m = JuMP.Model(with_optimizer(BARON.Optimizer, PrLevel=0, MaxTime=120))
        @variable(m, x[1:N])
        
        # Add objective
        @objective(m, Min, Base.invokelatest(eval(obj_expr), x))
        
        # Add upper/lower bound constraints
        @constraint(m, lbs[1:N] .<= x .<= ubs[1:N])
        
        # Add constraints
        for constr in constr_expr
           @constraint(m, Base.invokelatest(eval(constr), x)>=0)
        end
        
        return m
    end
    
    # Create the expression for the objective depending on whether
    # it is linear or not
    Q = create_negative_definite_matrix(N)
    c = randn(N)
    obj_expr = linear_objective ? :((x) -> $(c)'*x) : :((x) -> -x'*$(Q)*x - $(c)'*x)
    
    tmp = [(Q, c)]
    
    # Create the expressions for the constraints
    constr_expr = []
    for i=1:M
        Q = create_negative_definite_matrix(N)
        c = randn(N)
        expr = :((x) -> -x'*$(Q)*x - $(c)'*x)
        push!(constr_expr, expr)
        push!(tmp, (Q, c))
    end
    
    #return obj_expr
    return create_gm_model(obj_expr, constr_expr), create_baron_model(obj_expr, constr_expr)
end

function solve_and_benchmark(gm, gb, df_results, N, M)
    
    local function solve_gm()
        OCTHaGOn.globalsolve!(gm)
        x = gm.soldict[:x][1:N]
        return gm.cost[end], x
    end
    
    local function solve_baron()
        optimize!(gb)
        x_ref = [JuMP.variable_by_name(gb, "x[$(i)]") for i=1:N]
        x = value.(x_ref)
        return JuMP.objective_value(gb), x
    end
    
    ts = time()
    gm_obj, xgm = solve_gm()
    gm_time = time()-ts
    
    ts = time()
    baron_obj, xb = solve_baron()
    baron_time = time()-ts
    
    df_tmp = DataFrame(
        "n" => N,
        "m" => M,
        "gm" => gm_obj, 
        "baron" => baron_obj,
        "diff" => gm_obj-baron_obj,
        "gm_time" => gm_time,
        "ba_time" => baron_time
    )
    
    append!(df_results, df_tmp)
    
    return (gm_obj, xgm), (baron_obj, xb)
end

# Create the output path if not exists
output_path = "dump/qp/"
Base.Filesystem.mkpath(output_path)

df_results = DataFrame()

# Solve negative-definite QP with different 
# number of variables (N) and different number 
# of constraints (M)
for N=[5, 10, 20, 30, 40, 50, 60, 70]
    for M=[2, 5, 10, 15]
        try
            println("Solving with (N, M)=($(N),$(M))")
            gm, gb = create_noncvx_qp_models(N, M;seed=1)
            (gm_obj, xgm), (baron_obj, xb) = solve_and_benchmark(gm, gb, df_results, N, M)
            try
                CSV.write(output_path*"benchmark2.csv", df_results)
            catch
                println("Couldn't write to CSV")
            end
        catch
            println("Error solving (N, M)=($(N),$(M))")
            println(catch_backtrace())
        end
    end
end
    
print(df_results)

#OCTHaGOn.print_feas_gaps(gm)
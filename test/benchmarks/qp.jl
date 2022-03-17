
include("../load.jl")

using LinearAlgebra, Random, BARON, Pkg, Dates

EQUALITIES = false
REGRESSION = false
REPAIR = true 
OPT_SAMPLING = false

#algs = ["SVM", "CART", "GBM"]#, "CART", "OCT"

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
function create_noncvx_qp_models(n_vars, n_constr, algs, linear_objective=true; seed=123)
    
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
        
        #hashes = Set([])

        # Add constraints
        for constr in constr_expr
            OCTHaGOn.add_nonlinear_constraint(gm, constr, vars=x[1:N], expr_vars=[x[1:N]], alg_list=algs, equality=EQUALITIES, regression=REGRESSION)
            #constr_hash = bytes2hex(sha1("$(repr(constr))$(EQUALITIES)"))
            #push!(hashes, constr_hash)
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
            if EQUALITIES
                @constraint(m, Base.invokelatest(eval(constr), x)==0)
            else 
                @constraint(m, Base.invokelatest(eval(constr), x)>=0)
            end
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

function solve_and_benchmark(gm, gb, df_results, N, M, algs)
    
    local function solve_gm()
        OCTHaGOn.globalsolve!(gm; repair=REPAIR, opt_sampling=OPT_SAMPLING)
        x = gm.soldict[:x][1:N]
        return gm.cost[end], x
    end
    
    local function solve_baron(gm_hash)

        baron_output_path = "dump/qp/baron/baron2.csv"

        try
            df = DataFrame(CSV.File(baron_output_path))
        catch 
            df = DataFrame()
        end

        d = Dict(row["hash"] => row["obj"] for row in eachrow(df))

        obj = nothing
        x = nothing
        # If the same problem has been solved by baron,
        # load the results from file
        if haskey(d, gm_hash)
            obj = d[gm_hash]
            println("Loading baron solution from file")
        else
            optimize!(gb)
            x_ref = [JuMP.variable_by_name(gb, "x[$(i)]") for i=1:N]
            x = value.(x_ref)
            obj = JuMP.objective_value(gb)

            df = append!(df, DataFrame("obj" => obj, "hash" => gm_hash))
            try
                CSV.write(baron_output_path, df)
            catch
                println("Couldn't write to CSV1")
            end 
        end
        obj, x
    end
    #println("Calculating hash")
    gm_hash = OCTHaGOn.calculate_hash(gm)

    ts = time()
    baron_obj, xb = solve_baron(gm_hash)
    baron_time = time()-ts
    println("Baron solution: $(baron_obj)")

    ts = time()
    gm_obj, xgm = solve_gm()
    gm_time = time()-ts
    

    df_tmp = DataFrame(
        "n" => N,
        "m" => M,
        "gm" => gm_obj, 
        "baron" => baron_obj,
        "diff" => gm_obj-baron_obj,
        "gm_time" => gm_time,
        "ba_time" => baron_time,
        "algs" => "[\""*join(algs, "\",\"")*"\"]"
    )
    
    append!(df_results, df_tmp)
    
    return (gm_obj, xgm), (baron_obj, xb)
end

# Create the output path if not exists
output_path = "dump/qp/"
Base.Filesystem.mkpath(output_path)
suffix = Dates.format(Dates.now(), "YY-mm-dd_HH-MM-SS")

df_results = DataFrame()

# gm, gb = create_noncvx_qp_models(5, 1, ["SVM"];seed=1)
# solve_and_benchmark(gm, gb, df_results, 5, 1, ["SVM"])

# Solve negative-definite QP with different 
# number of variables (N) and different number 
# of constraints (M)
for N=[5, 10, 20, 30, 40, 50, 60, 70]
    for M=[1, 5, 10, 15]
        for algs in [["MLP"],["CART"],["SVM"],["GBM"], ["GBM", "CART"], ["GBM", "SVM", "CART"], ["GBM", "CART", "MLP"], ["GBM", "SVM", "CART", "MLP"]]#["SVM"],["CART"], ["GBM", "SVM", "CART"], ["OCT"], ["GBM", "SVM", "CART", "OCT"]]
            try
                algs = filter(x-> (x ∉ ["CART","OCT"]) || (!REGRESSION), algs)
                if length(algs) == 0 continue end

                println("Solving with (N, M)=($(N),$(M))")
                gm, gb = create_noncvx_qp_models(N, M, algs;seed=1)
                (gm_obj, xgm), (baron_obj, xb) = solve_and_benchmark(gm, gb, df_results, N, M, algs)
                try
                    
                    csv_path = output_path*"benchmark$(suffix).csv"
                    #println(csv_path)
                    CSV.write(csv_path, df_results)
                catch
                    println("Couldn't write to CSV")
                end
            catch
                println("Error solving (N, M)=($(N),$(M))")
                println(stacktrace(catch_backtrace()))
            end
            #break
            #print(df_results)
            #return;
        end
    end
end
print(df_results)

#OCTHaGOn.print_feas_gaps(gm)
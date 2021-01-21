# include("../load.jl")
using StatsBase

function random_qp(dims::Int64, nconstrs::Int64, sparsity=dims, solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, -1 <= x[1:dims] <= 1)
    sparse_vars = randperm(dims)[1:sparsity]
    m[:A] = rand(dims, sparsity).*2 .- 1
    m[:b] = rand(dims)
    m[:C] = rand(nconstrs, dims).*2 .- 1
    m[:d] = rand(nconstrs)
    @constraint(m, m[:C]*x .>= d)
    @objective(m, Min, sum((A*x[sparse_vars] - b).^2))
    return m
end

function gmify(m::JuMP.Model)
    @variable(m, obj)
    @objective(m, Min, obj)
    gm = GlobalModel(model = m)
    add_nonlinear_constraint(gm, :(x -> sum(($(m[:A])*x[$(sparse_vars)] - $(m[:b])).^2)), vars = m[:x][sparse_vars], 
                                    dependent_var = obj, name = "qp") 
    return gm
end

# Initializing, and solving via Ipopt
m = random_qp(10, 5, 4)
optimize!(m)
mcost = JuMP.getobjectivevalue(m)
msol = getvalue.(m[:x])

# Trying thresholding method 
using StatsBase
gm = gmify(m)
bbl = gm.bbls[1]
uniform_sample_and_eval!(gm)
uppers = []
lowers = []
actual = []
threshold = quantile(bbl.Y, 0.25)
learn_constraint!(gm, threshold=threshold)
add_tree_constraints!(gm)
optimize!(gm)

""" 
    find_leaf_of_soln(bbl::BlackBoxLearner)

Find leaf of previous solution via binary variables. 
"""
function find_leaf_of_soln(bbl::BlackBoxLearner)
    leaf_in = 0
    for (leaf, var) in bbl.leaf_variables
        if getvalue(var) == 1
            leaf_in = leaf
    end
    @assert leaf_in != 0
    return leaf_in
end
@assert leaf_in != 0
#UL_data for the leaf
(α0, α), (β0, β) = bbl.ul_data[end][leaf_in]
push!(uppers, α0 + sum(α .* getvalue.(bbl.vars)))
push!(lowers, β0 + sum(β .* getvalue.(bbl.vars)))
push!(actual, bbl(solution(gm)))

# Getting new data

leaf_neighbors = findall(x -> x == leaves[end], leaves);
df = DataFrame([Float64 for i in vks], vks)
build_knn_tree(bbl);
idxs, dists = find_knn(bbl, k=k);
leaves = IAI.apply(bbl.learners[end], bbl.X);
leaf_neighbors = randperm(MersenneTwister(1234), findall(x -> x == leaf_in, leaves));
count = 1
while size(df, 1) <= get_param(bbl, :n_samples)/2
    signs = [leaves[i] == leaf_in for i in idxs[count]]
    if length(signs) > sum(signs) > 0 
        inleaf = idxs[count][findall(x -> x == 1, signs)]
        outleaf = idxs[count][findall(x -> x != 1, signs)]
        for i in inleaf
            for j in outleaf
                np = (Array(bbl.X[i, :]) + Array(bbl.X[j, :]))./2
                push!(df, np)
            end
        end
    end
    count += 1
end
end

function knn_outward(bbl::BlackBoxClassifier, leaf, leaf_neighbors::Array; k::Int64 = 10, tighttol = 1e-5)
    vks = string.(bbl.vars)
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    feas_class = classify_patches(bbl, idxs);
    negatives = findall(x -> x .< 0, bbl.Y) # TODO: improve sign checking. 
    if !isnothing(sample_idxs)
        negatives = intersect(negatives, sample_idxs)
    end
    for i = 1:length(negatives) # This loop is for making sure that every possible root is sampled only once.
        if feas_class[negatives[i]] == "mixed"
            nodes = [idxs[negatives[i]][j] for j=1:length(idxs[negatives[i]]) 
                            if (bbl.Y[idxs[negatives[i]][j]] >= 0 && dists[negatives[i]][j] >= tighttol)]
            push!(nodes, negatives[i])
            np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
            append!(df, np)
        end
    end
    return df
end


            # df = knn_sample(bbl, k = length(bbl.vars), tighttol = get_param(gm, :tighttol), sample_idxs = leaf_neighbors)
            # last_sample = bbl.knn_tree.data[end]
            # println("New_samples: ", size(df))
            # if size(df, 1) > 0
            #     eval!(bbl, df)
            # #     weights = reweight(normalized_data(bbl))
            # #     learn_constraint!(bbl, sample_weight = weights)
            #     clear_tree_constraints!(gm, bbl)
            #     add_tree_constraints!(gm, bbl)
            # #     scatter!(df[:, string(vars[1])], df[:, string(vars[2])], color = :red)
            # end
            # LEARN TREE ON DF AS WELL AS ORIGINAL DATA IN THE LEAF. 
            # REMOVE ORIGINAL CONSTRAINT, RETRAIN WITH THE NEW DATA
            # weights = reweight(normalized_data(bbl), mag)
            # Learn greedy SVM approx locally
            # samp = 30
            # idxs, dists = OCT.knn(bbl.knn_tree, [last_sample], samp)
            # β0, β = svm(Matrix(bbl.X[idxs[1], :]), bbl.Y[idxs[1]])
            leaf_var = bbl.leaf_variables[leaves[end]]
            M = 1e5
            gfn = x -> ForwardDiff.gradient(bbl.f, x)
            varmap = get_varmap(bbl.expr_vars, bbl.vars)        
            inp = deconstruct(solution(gm), bbl.vars, varmap)
            grad = gfn(inp[1]...)    
            grad = [grad[v[2]] for v in varmap]
            constr = @constraint(gm.model, 0 <= sum(grad .* bbl.vars) + M * (1 - leaf_var))
            push!(bbl.mi_constraints, constr)
        end




gmsol = getvalue.(gm.model[:x])
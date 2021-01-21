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
    @constraint(m, m[:C]*x .>= m[:d])
    @objective(m, Min, sum((m[:A]*x[sparse_vars] - m[:b]).^2))
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

""" 
    find_leaf_of_soln(bbl::BlackBoxLearner)

Find leaf of previous solution via binary variables. 
"""
function find_leaf_of_soln(bbl::BlackBoxLearner)
    if !bbl.equality
        leaf_in = 0
        for (leaf, var) in bbl.leaf_variables
            if getvalue(var) == 1
                leaf_in = leaf
            end
        end
        @assert leaf_in != 0
        return leaf_in
    else
        leaf_in = []
        for (leaf, var) in bbl.leaf_variables
            if getvalue(var) == 1
                push!(leaf_in, leaf)
            end
        end
        @assert length(leaf_in) == 2
        return leaf_in
    end
end

"""
    knn_outward_from_leaf(bbl::BlackBoxLearner, leaf_in::Int64 = find_leaf_of_soln(bbl))

Binary knn samples from current leaf to surrounding leaves.   
"""
function knn_outward_from_leaf(bbl::BlackBoxLearner, leaf_in::Int64 = find_leaf_of_soln(bbl))
    leaves = IAI.apply(bbl.learners[end], bbl.X);
    leaf_neighbors = findall(x -> x == leaf_in, leaves);
    df = DataFrame([Float64 for i in vks], vks)
    build_knn_tree(bbl);
    idxs, dists = find_knn(bbl, k=k);
    leaf_neighbors = findall(x -> x == leaf_in, leaves)
    leaf_neighbors = leaf_neighbors[randperm(MersenneTwister(1234), length(leaf_neighbors))]; # Shuffling pre while loop
    ct = 1
    while size(df, 1) <= get_param(bbl, :n_samples)/2 && ct <= length(leaf_neighbors)
        neighbor = leaf_neighbors[ct] # sampling from inside outward
        stranger_idxs = findall(i -> leaves[i] != leaf_in, idxs[neighbor])
        for idx in stranger_idxs
            dist = dists[neighbor][idx]
            if dist >= get_param(gm, :tighttol)
                np = (Array(bbl.X[neighbor, :]) + Array(bbl.X[idxs[neighbor][idx], :]))./2
                push!(df, np)
            end
        end
        ct += 1
    end
    return df
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
actuals = []
threshold = quantile(bbl.Y, 0.25)
learn_constraint!(gm, threshold=threshold)
add_tree_constraints!(gm)
optimize!(gm)



leaf_in = find_leaf_of_soln(bbl)
#UL_data for the leaf
(α0, α), (β0, β) = bbl.ul_data[end][leaf_in]
push!(uppers, α0 + sum(α .* getvalue.(bbl.vars)))
push!(lowers, β0 + sum(β .* getvalue.(bbl.vars)))
push!(actuals, bbl(solution(gm))[1])
# Resample
df = knn_outward_from_leaf(bbl, leaf_in)
eval!(bbl, df)

# Do something about uppers, lowers and actuals. 
if actuals[end] <= lowers[end]
    threshold = lowers[end]
elseif lowers[end] <= actuals[end] <= uppers[end]
    threshold = (uppers[end] - actuals[end])/2 + actuals[end]
elseif actuals[end] >= uppers[end]
    threshold = threshold
    push!(bbl.learners, bbl.learners[end])
    push!(bbl.ul_data, ul_boundify(bbl.learners[end], bbl.X, bbl.Y))
end


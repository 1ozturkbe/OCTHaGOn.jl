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
# Find the leaf we are in via binary variables
leaf_in = 0
for (leaf, var) in bbl.leaf_variables
    if getvalue(var) == 1
        leaf_in = leaf
    end
end
@assert leaf_in != 0
#UL_data for the leaf
(α0, α), (β0, β) = bbl.ul_data[end][leaf_in]
push!(uppers, α0 + sum(α .* getvalue.(bbl.vars)))
push!(lowers, β0 + sum(β .* getvalue.(bbl.vars)))
push!(actual, bbl(solution(gm)))

if actual[end] >= 

leaf = getvalue.(collect(bbl.leaf_variables))


gmsol = getvalue.(gm.model[:x])
#  Let's build tree structure (borrowed from AbstractTrees.jl)
mutable struct BinaryNode{T}
    idx::T
    parent::BinaryNode{T}
    left::BinaryNode{T}
    right::BinaryNode{T}
    a::Array 
    b::Real
    label::Any

    # Root constructor
    BinaryNode{T}(idx) where T = new{T}(idx)
    # Child node constructor
    BinaryNode{T}(idx, parent::BinaryNode{T}) where T = new{T}(idx, parent)
end
BinaryNode(idx) = BinaryNode{typeof(idx)}(idx)

function leftchild(idx, parent::BinaryNode)
    !isdefined(parent, :left) || error("left child is already assigned")
    node = typeof(parent)(idx, parent)
    parent.left = node
end
function rightchild(idx, parent::BinaryNode)
    !isdefined(parent, :right) || error("right child is already assigned")
    node = typeof(parent)(idx, parent)
    parent.right = node
end

""" 
    children(node::BinaryNode)

Returns (left, right) children of a BinaryNode.
"""
function children(node::BinaryNode)
    if isdefined(node, :left)
        if isdefined(node, :right)
            return (node.left, node.right)
        end
        return (node.left,)
    end
    isdefined(node, :right) && return (node.right,)
    return ()
end

""" Better node printing."""
printnode(io::IO, node::BinaryNode) = print(io, node.idx)

""" 
Returns a-coefficients of hyperplane split on BinaryNode. 
"""
function get_split_weights(bn::BinaryNode)
    return bn.a
end

""" 
Returns b-value of hyperplane split on BinaryNode. 
"""
function get_split_threshold(bn::BinaryNode)
    return bn.b
end

""" 
    is_leaf(bn::BinaryNode)

Checks whether a BinaryNode is a leaf of the tree. 
"""
function is_leaf(bn::BinaryNode)
    return !isdefined(bn, :left) && !isdefined(bn, :right)
end

"""
    get_classification_label(bn::BinaryNode)

Returns the classification label of a leaf BinaryNode. 
"""
function get_classification_label(bn::BinaryNode)
    is_leaf(bn) || throw(OCTException("Cannot get the classification label of node $(bn.idx), " * 
                        "since it is not a leaf node."))
    return bn.label
end

"""
    MIOTree_defaults(kwargs...)

Contains default MIOTree parameters, and modifies them with kwargs.
"""
function MIOTree_defaults(kwargs...)
    d = Dict(:random_seed => 1,
        :max_depth => 5,
        :cp => 1e-6,
        :minbucket => 0.01,
        :max_time => 20) # in seconds
    if !isempty(kwargs)
        for (key, value) in kwargs
            set_param(d, key, value)
        end
    end
    return d
end

""" 
    generate_binary_tree(root::BinaryNode, max_depth::Int)

Returns a dict, describing the nodes in a binary tree of max_depth.
"""
function generate_binary_tree(root::BinaryNode, max_depth::Int)
    # Constructing a binary tree of given max_depth
    root = BinaryNode(1)
    levels = Dict{Int64, Array{BinaryNode}}(0 => [root])
    for i = 1:max_depth
        idxs = levels[i-1][end].idx .+ collect(1:2^i)
        levels[i] = BinaryNode[]
        for j = 1:length(levels[i-1])
            push!(levels[i], leftchild(idxs[2*j-1], levels[i-1][j]))
            push!(levels[i], rightchild(idxs[2*j], levels[i-1][j]))
        end
    end
    return levels
end
    
@with_kw mutable struct MIOTree
    model::JuMP.Model
    root::BinaryNode = BinaryNode(0)
    params::Dict = MIOTree_defaults()
    levels::Dict{Int, Array{BinaryNode}} = 
        generate_binary_tree(root, params.max_depth)
    
    function MIOTree(model::JuMP.Model; kwargs...)
        root = BinaryNode(0)
        params = MIOTree_defaults(kwargs...)
        return new(model, root, params,
            generate_binary_tree(root, get_param(params, :max_depth)))
    end
    MIOTree(solver::MathOptInterface.OptimizerWithAttributes; kwargs...) = 
        MIOTree(Model(solver); kwargs...)
end

get_param(mt::MIOTree, sym::Symbol) = get_param(mt.params, sym)

function set_param(mt::MIOTree, sym::Symbol, val::Any) 
    set_param(mt.params, sym, val)
    if sym == :max_depth
        mt.root = BinaryNode(0)
        mt.levels = generate_binary_tree(mt.root, val) 
        # TODO: decide whether to regenerate mt.model
    elseif sym == :max_time
        set_optimizer(mt, with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0, CPX_PARAM_TILIM = val))
    # TODO: complete with the rest of the parameters. Test initialization. 
    end
end

JuMP.set_optimizer(mt::MIOTree, solver) = JuMP.set_optimizer(mt.model, solver)

JuMP.optimize!(mt::MIOTree) = JuMP.optimize!(mt.model)

function generate_tree_model(tree::MIOTree, X::Matrix, Y::Array; solver = CPLEX_SILENT)
    n_samples, n_vars = size(X)

    # Reference minimal parameters
    max_depth = get_param(tree, :max_depth)
    levels = tree.levels
    n_nodes = sum(2^i for i=0:max_depth)
    n_leaves = 2^max_depth
    n_splits = n_nodes - n_leaves
    min_points = Int(ceil(get_param(tree, :minbucket) * n_samples))
    unique_Y = sort(unique(Y)) # The potential classes are sorted. 
    k = length(unique(Y))
    k != 2 && @info("Detected $(k) unique classes.")

    @variable(tree.model, -1 <= a[1:n_splits, 1:n_vars] <= 1)

    @variable(tree.model, -1 <= b[1:n_splits] <= 1)
    @variable(tree.model, d[1:n_splits], Bin)
    @variable(tree.model, s[1:n_splits, 1:n_vars], Bin) # Binary variables for complexity penalty
    @constraint(tree.model, -d .<= b)
    @constraint(tree.model, b .<= d)
    @constraint(tree.model, -s .<= a)
    @constraint(tree.model, a .<= s)
    @constraint(tree.model, [j=1:n_vars], s[:,j] .<= d[:])
    @constraint(tree.model, [i=1:n_splits], sum(s[i, :]) >= d[i])
    
    # Enforcing each point to one leaf
    @variable(tree.model, z[1:n_samples, 1:n_leaves], Bin)
    @constraint(tree.model, [i=1:n_leaves], sum(z[:, i]) == 1)

    # Making sure that variables are properly binned. 
    @variable(tree.model, ckt[1:n_leaves, 1:k], Bin)  # Class at leaf
    @variable(tree.model, Nt[1:n_leaves] >= 0)       # Total number of points at leaf
    @variable(tree.model, lt[1:n_leaves], Bin)       # Whether or not a leaf is occupied
    @variable(tree.model, Nkt[1:n_leaves, 1:k] >= 0) # Number of points of at leaf with class k

    @constraint(tree.model, [i = 1:n_leaves], Nt[i] == sum(z[:, i])) # Counting number of points in a leaf. 
    @constraint(tree.model, [i = 1:n_leaves], sum(ckt[i, :]) == lt[i]) # Making sure a class is only assigned if leaf is occupied.
    for kn = 1:k
        # Number of values of each class
        @constraint(tree.model, [i=1:n_leaves], Nkt[i, kn] == 
                    sum(z[l, i] for l = 1:n_samples if Y[l] == kn))
    end

    # Loss function
    @variable(tree.model, Lt[1:n_leaves] >= 0)
    @constraint(tree.model, [i = 1:n_leaves, j = 1:k], Lt[i] >= Nt[i] - Nkt[i,j] - n_samples * (1-ckt[i,j]))
    @constraint(tree.model, [i = 1:n_leaves, j = 1:k], Lt[i] <= Nt[i] - Nkt[i,j] + n_samples * ckt[i,j])

    for depth = 1:max_depth-1
        for nd in levels[depth]
            # Enforcing that branches can only apply split if parents apply splits
            @constraint(tree.model, d[nd.idx] <= d[nd.parent.idx])
        end
    end

    mu = 0.005
    for j = 1:n_leaves
        # Enforcing minbucket # TODO: fix minbucket
        # @constraint(tree.model, [i=1:n_samples], z[i, j] <= lt[j])
        # @constraint(tree.model, sum(z[:, j]) >= min_points*lt[j])
        # Enforcing hyperplane splits
        for i=1:n_samples
            nd = levels[max_depth][j]
            for _ = 1:max_depth # while loop wasn't working... scoping is hard
                if nd.idx == nd.parent.left.idx
                    @constraint(tree.model, sum(a[nd.parent.idx, :] .* X[i, :]) + mu <= b[nd.parent.idx] + (2+mu)*(1-z[i,j])) 
                elseif nd.idx == nd.parent.right.idx
                    @constraint(tree.model, sum(a[nd.parent.idx, :] .* X[i, :]) >= b[nd.parent.idx] - 2*(1-z[i,j])) 
                else
                    throw(OCTException("Node backtracking failed for some reason."))
                end
                try 
                    nd = nd.parent
                catch err
                    if isa(err, UndefRefError)
                        break
                    else
                        throw(OCTException("Node parenting failed unexpectedly."))
                        break
                    end
                end
            end
        end
    end

    # Objective function: misclassification error + complexity
    @objective(tree.model, Min, 1/n_samples * sum(Lt) + get_param(tree, :cp) * sum(s[:,:]))
    return
end


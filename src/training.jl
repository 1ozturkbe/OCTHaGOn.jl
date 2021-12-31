#  Let's build tree structure (borrowed from AbstractTrees.jl)
@with_kw mutable struct BinaryNode{T}
    idx::T
    parent::Union{BinaryNode{T}, Nothing} = nothing
    left::Union{BinaryNode{T}, Nothing} = nothing 
    right::Union{BinaryNode{T}, Nothing} = nothing
    a::Union{Array, Nothing} = nothing 
    b::Union{Real, Nothing} = nothing
    label::Any = nothing
end
BinaryNode(idx) = BinaryNode{typeof(idx)}(idx = idx)
BinaryNode(idx, parent::BinaryNode) = BinaryNode{typeof(idx)}(idx = idx, parent = parent)

function leftchild(parent::BinaryNode, child::BinaryNode)
    isnothing(parent.left) || error("Left child of node $(parent.idx) is already assigned.")
    isnothing(child.parent) || error("Parent of node $(child.idx) is already assigned.")
    parent.left = child    
    child.parent = parent
    return
end

function rightchild(parent::BinaryNode, child::BinaryNode)
    isnothing(parent.right) || error("Right child of node $(parent.idx) is already assigned.")
    isnothing(child.parent) || error("Parent of node $(child.idx) is already assigned.")
    parent.right = child    
    child.parent = parent
    return
end

""" 
    children(node::BinaryNode)

Returns (left, right) children of a BinaryNode.
"""
function children(node::BinaryNode)
    if !isnothing(node.left)
        if !isnothing(node.right)
            return (node.left, node.right)
        end
        return (node.left,)
    end
    !isnothing(node.right) && return (node.right,)
    return ()
end

""" Returns all "younger" relatives of a BinaryNode. """
function alloffspring(nd::BinaryNode)
    offspr = [child for child in children(nd)]
    queue = [child for child in children(nd)]
    while !isempty(queue)
        nextnode = pop!(queue)
        if !is_leaf(nextnode)
            for child in children(nextnode)
                push!(offspr, child)
                push!(queue, child)
            end
        else
            for child in children(nextnode)
                push!(offspr, child)
            end
        end
    end
    return offspr
end

"""
    delete_children!(bn::BinaryNode)

"Deletes" the children of BinaryNode by pruning and returning the nodes.   
"""
function delete_children!(bn::BinaryNode)
    allchildren = children(bn)
    bn.left = nothing
    bn.right = nothing
    for child in allchildren
        child.parent = nothing
    end
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
    return isnothing(bn.left) && isnothing(bn.right)
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
    d = Dict(:max_depth => 5,
        :cp => 1e-6,
        :minbucket => 0.01) # in seconds
    if !isempty(kwargs)
        for (key, value) in kwargs
            set_param(d, key, value)
        end
    end
    return d
end

""" 
    generate_binary_tree(root::BinaryNode, max_depth::Int)

Returns all nodes and leaves of a binary tree.
"""
function generate_binary_tree(root::BinaryNode, max_depth::Int)
    levs = Dict{Int64, Array{BinaryNode}}(0 => [root])
    nodes = [root]
    for i = 1:max_depth
        idxs = levs[i-1][end].idx .+ collect(1:2^i)
        levs[i] = BinaryNode[]
        for j = 1:length(levs[i-1])
            lc = BinaryNode(idxs[2*j-1])
            leftchild(levs[i-1][j], lc)
            push!(levs[i], lc)
            push!(nodes, lc)
            rc = BinaryNode(idxs[2*j])
            rightchild(levs[i-1][j], rc)
            push!(levs[i], rc)
            push!(nodes, rc)
        end
    end
    return nodes, levs[max_depth]
end
    
@with_kw mutable struct MIOTree
    model::JuMP.Model = JuMP.Model()
    root::BinaryNode = BinaryNode(1)
    params::Dict = MIOTree_defaults()
    classes::Union{Nothing, Array{Any}} = nothing
    nodes::Union{Nothing, Array{BinaryNode}} = nothing
    leaves::Union{Nothing, Array{BinaryNode}} = nothing
end

function build_MIOTree(model::JuMP.Model; kwargs...)
    mt = MIOTree(model = model)
    for item in kwargs
        if item.first == :root
            mt[item.first] = item.second
        elseif item.first in keys(mt.params)
            set_param(mt.params, item.first, item.second)
        end
    end
    mt.nodes, mt.leaves = generate_binary_tree(mt.root, get_param(mt, :max_depth))
    return mt
end
build_MIOTree(model::JuMP.Model) = build_MIOTree(model)
build_MIOTree(; kwargs...) = build_MIOTree(JuMP.Model(); kwargs...)
build_MIOTree(solver::MathOptInterface.OptimizerWithAttributes; kwargs...) =  
    build_MIOTree(JuMP.Model(solver); kwargs...)

get_param(mt::MIOTree, sym::Symbol) = get_param(mt.params, sym)

"""
    set_param(mt::MIOTree, sym::Symbol, val::Any) 

Sets parameters for the MIOTree. Note that parameters that change
tree structure take effect IMMEDIATELY on .nodes, .leaves,
but NOT ON .model.
Model must be regenerated by user via generate_tree_model.
"""
function set_param(mt::MIOTree, sym::Symbol, val::Any) 
    set_param(mt.params, sym, val)
    if sym == :max_depth
        mt.root = BinaryNode(mt.root.idx)
        mt.nodes, mt.leaves = generate_binary_tree(mt.root, val) 
    end
end

JuMP.set_optimizer(mt::MIOTree, solver) = JuMP.set_optimizer(mt.model, solver)

JuMP.optimize!(mt::MIOTree) = JuMP.optimize!(mt.model)

"""
    generate_tree_model(mt::MIOTree, X::Matrix, Y::Array)

Generates a JuMP MIO model of the tree classifier, with data X and Y.
"""
function generate_tree_model(mt::MIOTree, X::Matrix, Y::Array)
    if !isempty(mt.model.obj_dict)
        mt.model = JuMP.Model() # TODO: find a way to add solver in here.
    end
    n_samples, n_vars = size(X)

    # Reference minimal parameters
    max_depth = get_param(mt, :max_depth)
    n_nodes = length(mt.nodes)
    n_leaves = length(mt.leaves)
    n_splits = n_nodes - n_leaves
    nd_idxs = [nd.idx for nd in mt.nodes] # Node indices
    lf_idxs = [lf.idx for lf in mt.leaves] # Leaf indices
    sp_idx = [idx for idx in nd_idxs if idx ∉ lf_idxs]
    min_points = get_param(mt, :minbucket)
    if !isa(min_points, Int) && 0 <= min_points <= 1
        min_points = Int(ceil(min_points * n_samples))
    else
        throw(OCTException("Minbucket parameter must be between 0-1 or an integer!"))
    end
    mt.classes = sort(unique(Y)) # The potential classes are sorted. 
    k = length(mt.classes)
    k != 2 && @info("Detected $(k) unique classes.")

    @variable(mt.model, -1 <= a[1:n_splits, 1:n_vars] <= 1)
    @variable(mt.model, 0 <= abar[1:n_splits, 1:n_vars])
    @variable(mt.model, -1 <= b[1:n_splits] <= 1)
    @variable(mt.model, d[1:n_splits], Bin)
    @variable(mt.model, s[1:n_splits, 1:n_vars], Bin) # Binary variables for complexity penalty
    @constraint(mt.model, -b .<= d)
    @constraint(mt.model, b .<= d)
    @constraint(mt.model, -a .<= s)
    @constraint(mt.model, a .<= s)
    @constraint(mt.model, a .<= abar)
    @constraint(mt.model, -a .<= abar)
    @constraint(mt.model, [j=1:n_vars], s[:,j] .<= d[:])
    @constraint(mt.model, [i=1:n_splits], sum(s[i, :]) >= d[i])
    @constraint(mt.model, [i=1:n_splits, j=1:n_vars], a[i,j] <= s[i,j])
    @constraint(mt.model, [i=1:n_splits, j=1:n_vars], -s[i,j] <= a[i,j])
    
    # Enforcing each point to one leaf
    @variable(mt.model, z[1:n_samples, 1:n_leaves], Bin)
    @constraint(mt.model, [i=1:n_samples], sum(z[i, :]) == 1)

    # Making sure that variables are properly binned. 
    @variable(mt.model, ckt[1:k, 1:n_leaves], Bin)  # Class at leaf
    @variable(mt.model, Nt[1:n_leaves] >= 0)       # Total number of points at leaf
    @variable(mt.model, lt[1:n_leaves], Bin)       # Whether or not a leaf is occupied
    @variable(mt.model, Nkt[1:k, 1:n_leaves] >= 0) # Number of points of at leaf with class k

    @constraint(mt.model, [i = 1:n_leaves], Nt[i] == sum(z[:, i])) # Counting number of points in a leaf. 
    @constraint(mt.model, [i = 1:n_leaves], sum(ckt[:, i]) == lt[i]) # Making sure a class is only assigned if leaf is occupied.
    for kn = 1:k
        # Number of values of each class
        @constraint(mt.model, [i=1:n_leaves], Nkt[kn, i] == 
                    sum(z[l, i] for l = 1:n_samples if Y[l] == mt.classes[kn]))
    end

    # Loss function
    @variable(mt.model, Lt[1:n_leaves] >= 0)
    @constraint(mt.model, [i = 1:n_leaves, j = 1:k], Lt[i] >= Nt[i] - Nkt[j, i] - n_samples * (1-ckt[j,i]))
    @constraint(mt.model, [i = 1:n_leaves, j = 1:k], Lt[i] <= Nt[i] - Nkt[j, i] + n_samples * ckt[j,i])

    @constraint(mt.model, sum(abar[1, :]) <= d[1])
    for nd in mt.nodes
        if !is_leaf(nd) && !isnothing(nd.parent)
            @constraint(mt.model, d[nd.idx] <= d[nd.parent.idx])
            @constraint(mt.model, sum(abar[nd.idx, :]) <= d[nd.idx])
        end
    end

    mu = 5e-5
    for j = 1:n_leaves
        lf = mt.leaves[j]
        # Enforcing minbucket 
        @constraint(mt.model, [i=1:n_samples], z[i, j] <= lt[j])
        @constraint(mt.model, sum(z[:, j]) >= min_points*lt[j])
        # Enforcing hyperplane splits
        for i=1:n_samples
            nd = lf
            for _ = 1:max_depth # while loop wasn't working... scoping is hard
                if nd.idx == nd.parent.left.idx
                    @constraint(mt.model, sum(a[nd.parent.idx, :] .* X[i, :]) + mu <= b[nd.parent.idx] + (2+mu)*(1-z[i,j])) 
                elseif nd.idx == nd.parent.right.idx
                    @constraint(mt.model, sum(a[nd.parent.idx, :] .* X[i, :]) >= b[nd.parent.idx] - 2*(1-z[i,j])) 
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
    @objective(mt.model, Min, 1/n_samples * sum(Lt) + get_param(mt, :cp) * sum(s[:,:]))
    return
end

""" 
Returns the misclassification error of the MIOTree.
"""
function score(mt::MIOTree)
    return sum(JuMP.getvalue.(mt.model[:Lt]))/size(mt.model[:z],1)
end

"""
Returns the number of nonzero hyperplane coefficients of the MIOTree. 
"""
function complexity(mt::MIOTree)
    return sum(JuMP.getvalue.(mt.model[:s]))
end

"""
    populate_nodes!(mt::MIOTree)

Populates the nodes of the MIOTree using 
optimal solution of the MIO problem. 
"""
function populate_nodes!(mt::MIOTree)
    termination_status(mt.model) == MOI.OPTIMAL || 
        throw(OCTException("MIOTree must be trained before it can be pruned."))
    queue = BinaryNode[mt.root]
    leaves = BinaryNode[]
    m = mt.model
    as = getvalue.(m[:a])
    bs = getvalue.(m[:b])
    ckt = getvalue.(m[:ckt]) # Finding labels
    while !isempty(queue) # First populate the a,b hyperplane values
        nd = pop!(queue)
        if !is_leaf(nd)
            if any(!iszero(as[nd.idx, :]))
                nd.a = as[nd.idx, :]
                nd.b = bs[nd.idx]
                for child in children(nd)
                    push!(queue, child)
                end
            end
        end
    end
    for i = 1:length(mt.leaves) # Then populate the class values...
        class_values = findall(ckt[:, i] .== 1)
        if length(class_values) == 1
            mt.leaves[i].label = mt.classes[class_values[1]]
        elseif length(class_values) > 1
            throw(OCTException("Multiple classes assigned to node $(mt.leaves[i].idx)."))
        end
    end
    return
end

""" 
    prune!(mt::MIOTree)

Populates and prunes MIOTree, and updates .nodes and .leaves. 
See ```populate_nodes''' for 
how to populate nodes using optimal solution data. 
"""
function prune!(mt::MIOTree)
    populate_nodes!(mt)
    leaves = BinaryNode[]
    queue = BinaryNode[mt.root]
    while !isempty(queue)
        nd = pop!(queue)
        if !isnothing(nd.a) && any(nd.a != 0)
            for child in children(nd)
                push!(queue, child)
            end
        else
            alloffspr = alloffspring(nd)
            if isnothing(nd.label)
                alllabels = [nextnode.label for nextnode in alloffspr if !isnothing(nextnode.label)]
                if length(alllabels) == 1 
                    nd.label = alllabels[1]

                elseif length(alllabels) > 1
                    throw(OCTException("Too many labels below node $(nd.idx)! Bug!"))
                else
                    throw(OCTException("Missing labels below node $(nd.idx)! Bug!"))
                end
            end
            delete_children!(nd)
            push!(leaves, nd)
        end
    end
    mt.nodes = union([mt.root], alloffspring(mt.root))
    mt.leaves = leaves
    # Add checks here so that the number of branches + leaves == total number of nodes
    return
end
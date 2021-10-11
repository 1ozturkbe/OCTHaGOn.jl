# Parameters
random_seed = 1
max_depth = 5
cp = 1e-6
minbucket = 0.01
fast_num_support_restarts = 5
localsearch = true
ls_num_tree_restarts = 10
ls_num_hyper_restarts = 5

#  Let's build tree structure (borrowed from AbstractTrees.jl)
mutable struct BinaryNode{T}
    idx::T
    parent::BinaryNode{T}
    left::BinaryNode{T}
    right::BinaryNode{T}

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

# Returns (left, right) children of a BinaryNode
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

## Things that make printing prettier
printnode(io::IO, node::BinaryNode) = print(io, node.idx)

# Constructing a binary tree
n_nodes = sum(2^i for i=0:max_depth)
root = BinaryNode(1)
nodes = [root]
prev_level = [root]
for i = 1:max_depth
    idxs = prev_level[end].idx .+ collect(1:2^i)
    next_level = []
    for j = 1:length(prev_level)
        push!(next_level, leftchild(idxs[2*j-1], prev_level[j]))
        push!(next_level, rightchild(idxs[2*j], prev_level[j]))
    end
    append!(nodes, next_level)
    prev_level = next_level
end


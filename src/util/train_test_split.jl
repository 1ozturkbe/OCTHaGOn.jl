

function train_test_split(X::Matrix; at = 0.7)
    n = size(X,1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    X[train_idx,:], X[test_idx,:]
end

function train_test_split(X::DataFrame; at = 0.7)
    n = nrow(X)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    X[train_idx,:], X[test_idx,:]
end

function train_test_split(X::DataFrame, Y::Union{Array, Vector, BitVector}; at = 0.7)
    n = nrow(X)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    (X[train_idx,:], Y[train_idx]), (X[test_idx,:], Y[test_idx])
end

function train_test_split(X::Matrix, Y::Union{Array, Vector, BitVector}; at = 0.7)
    n = size(X, 1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    (X[train_idx,:], Y[train_idx]), (X[test_idx,:], Y[test_idx])
end

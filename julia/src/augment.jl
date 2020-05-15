using DataFrames

function normalize(X)
    # Normalizes each column by the standard normal
    n,p = size(X)
    X_n = zeros(n,p)
    means = [sum(X[:,i])/n for i=1:p];
    vars = [sum((X[:,i] - means[i]*ones(n,1)).^2)/n for i=1:p];
    for i=1:p
        X_n[:,i] = (X[:,i] - means[i]*ones(n,1))./vars[i]^(0.5);
    end
    return X_n, means, vars
end

function threshold(Y, val)
    # Thresholds Y and turns into binary categories
    return float(Y .>= val)
end

function augment(X, fns, fn_names=nothing)
    # Augmenting the data with a set of fn's, and returns associated functions
    # This method prefers that fns represent operations on columns, not matrices!
    n,p = size(X);
    augX = deepcopy(X);
    fn_array = [x -> x[:,i] for i=1:p];
    for i=1:size(fns,1)
        next = fns[i](X);
        if typeof(X) == DataFrame && fn_names
            names!(next, fn_names[i])
        end
        augX = hcat(augX, next);
        fn_array= vcat(fn_array, [fns[i]]);
    end
    return augX
end

function quad_fns(x)
    n,p = size(x);
    fns = [x->x[:,i].^2 for i = 1:p];
    fn_names = [];
    if typeof(x) == DataFrame
        fn_names = [string("x_", i, "^2") for i =1:p];
    end
    return fns, fn_names
end

function bilinear_fns(x)
    n,p = size(x);
    fn_arr = [x->x[:,i].^2 for i = 1:p];
    fn_names = [];
    if typeof(x) == DataFrame
        fn_names = [string("x_", i, "^2") for i =1:p];
    end
    for i = 1:p
        fn_arr = vcat(fn_arr, [x->x[:,i].*x[:,j] for j=i+1:p])
        if typeof(x) == DataFrame
            fn_names = vcat(fn_names, [string("x_", i,"*", "x_", j ) for j=i+1:p]);
        end
    end
    return fn_arr, fn_names
end

function sixth_order_even_fns(x)
    n,p = size(x);
    fn_arr = [x->x[:,i].^2 for i = 1:p];
    fn_arr = vcat(fn_arr,[x->x[:,i].^4 for i = 1:p]);
    fn_arr = vcat(fn_arr, [x->x[:,i].^6 for i = 1:p];)
    fn_names = [];
    if typeof(x) == DataFrame
        fn_names = [string("x_", i, "^2") for i =1:p];
        fn_names = vcat(fn_names,[string("x_", i, "^4") for i =1:p]);
        fn_names = vcat(fn_names, [string("x_", i, "^6") for i =1:p]);
    end
    for i = 1:p
        fn_arr = vcat(fn_arr, [x->x[:,i].^2 .* x[:,j].^4 for j=1:p])
        if typeof(x) == DataFrame
            fn_names = vcat(fn_names, [string("x_", i,"^2*", "x_", j, "^4*") for j=1:p]);
        end
    end
    return fn_arr, fn_names
end

function exps(x)
    n,p = size(x)
    return [x->exp.(x[:,i]) for i=1:p]
end

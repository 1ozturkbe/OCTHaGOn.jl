using Distributions

function reweight(X, x_sol, mag=1)
    """ Gaussian reweighting of existing data by proximity to previous solution.
    Arguments:
    - X: data
    - x_sol: solution array with same # of columns
    - mag: relative weighting, bigger is more.
    Returns:
    - weights: weights of X rows, by Euclidian distance
    """
    n_samples, n_features = size(X);
    mean = [sum(X[:,i])/n_samples for i=1:n_features];
    std = [sum((X[:,i]-ones(n_samples)* mean[i]).^2)/n_samples for i=1:n_features];
    distance = [sum((X[i,:] - mean).^2 ./std.^2) for i=1:n_samples];
    weights = exp.(-0.5*mag*distance);
    return weights
end

function resample(n_samples, lbs, ubs, prev_X, x_sol; mean_shift=0.75, std_shrink = 0.3)
    """ Gaussian resamples free variables depending on previous solution.
    Arguments:
    - n_samples: number of samples
    - lbs: lower bounds
    - bs: upper bounds
    - prev_X: previous data
    - x_sol: solution array with same # of columns as lbs/ubs
    - sigma: by how much to move the mean to new optimum, and to shrink the std.
    Returns:
    - new_samples
    """
    X = deepcopy(prev_X);
    prev_samples, n_features = size(prev_X);
    prev_mean = [sum(X[:,i])/prev_samples for i=1:n_features];
    prev_std = [sum((X[:,i]-ones(prev_samples)* prev_mean[i]).^2)/prev_samples for i=1:n_features];
    new_std = prev_std*(1-std_shrink);
    new_mean = mean_shift*x_sol + (1-mean_shift)*prev_mean;
    dist = MvNormal(new_mean, new_std);
    while size(X,1) < prev_samples + n_samples
        new_row = rand(dist)';
        if all(lbs .<= new_row) && all(ubs .>= new_row);
            X = vcat(X, Matrix(new_row));
        end
    end
    return X
end

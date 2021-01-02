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

# Sampling based on GaussianProcesses
#         plan = randomLHC(Int(round(get_param(bbf, :n_samples)*ratio)), n_dims);
#         random_samples = scaleLHC(plan,[(bbf.lbs[i], bbf.ubs[i]) for i in vks]);
#         μ, σ = predict_f(bbf.gp, random_samples');
#         cdf_0 = [Distributions.cdf(Distributions.Normal(μ[i], σ[i]),0) for i=1:size(random_samples,1)];
#          #TODO: add criterion for information as well (something like sortperm(σ))
#         # Sample places with high probability of being near boundary (0),
#         # but also balance feasibility ratio.
#         p = bbf.feas_ratio
# #         balance_fn = x -> -1*tan(2*atan(-0.5)*(x-0.5)) + 0.5
#         balance_fn = x -> -1/0.5^2*(x-0.5)^3 + 0.5;
#         indices = sortperm(abs.(cdf_0 .- balance_fn(p)));
#         samples = DataFrame(random_samples[indices[1:get_param(bbf, :n_samples)],:], vks);
#         eval!(bbf, samples);

# function predict(bbf::BlackBoxFunction, X::AbstractArray)
#     μ, σ = predict_f(bbf.gp, transpose(X))
#     return μ, σ
# end
#
# function optimize_gp!(bbf::BlackBoxFunction)
#     """ Optimizes a GaussianProcess over a BlackBoxFunction,
#     with adaptively changing kernel. """
# #         bbf.gp = ElasticGPE(length(bbf.idxs), # data
# #         mean = MeanConst(sum(bbf.Y)/length(bbf.Y)), logNoise = -10)
#     lbs = [bbf.lbs[key] for key in bbf.vks];
#     ubs = [bbf.ubs[key] for key in bbf.vks];
#     bbf.gp = GPE(transpose(Array(bbf.X)), bbf.Y, # data
#     MeanConst(sum(bbf.Y)/length(bbf.Y)),
#     SEArd(log.((ubs-lbs)./(2*sqrt(length(bbf.Y)))), -5.))
#     optimize!(bbf.gp); #TODO: optimize GP
#                        # Instead of regenerating at every run, figure out
#                        # how to update.
# end
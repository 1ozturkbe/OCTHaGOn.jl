# R Functions for design of experiments

using RCall

function ort_arrays(levels)
    # Based on this: https://support.sas.com/techsup/technote/ts723b.pdf
    R"library(DoE.base)"
    R"orth_arrays <- show.oas(nlevels=$levels,  showGRs=TRUE)"
    @rget orth_arrays
    return orth_arrays
end

function full_factorial_doe(levels)
    R"library(DoE.base)"
    R"factor_mat <- oa.design(nlevels=$levels, seed=100)"
    @rget factor_mat
    return factor_matcmd
end

function collapse(orth_array, nlevels)
    new_mat = zeros(size(orth_array,1), size(nlevels,1));
    for i=1:size(nlevels,1)
        for j = 1:size(orth_array,2)
            if maximum(orth_array[:,j]) == nlevels[i]
                new_mat[:,i] = orth_array[:,j];
                orth_array[:,j] = zeros(size(orth_array[:,j]));
                break
            end
        end
    end
    return new_mat
end

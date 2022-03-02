

# This set of functions is based entirely on the excellent
# scikit-learn package's method of calculating an ROC's area
# under the curve (AUC). The functions below produce the
# same output as scikit-learn.

function _binary_clf_curve(y_true, y_score)
    y_true = y_true .== 1       # make y_true a boolean vector
    desc_score_indices = sortperm(y_score, rev = true)

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = find(diff(y_score))
    threshold_idxs = push!(distinct_value_indices, length(y_score))

    tps = cumsum(y_true)[threshold_idxs]
    fps = threshold_idxs - tps
    return (fps, tps, y_score[threshold_idxs])
end



function _roc_curve(y_true, y_score)
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    fpr = fps/fps[end]
    tpr = tps/tps[end]
    return (fpr, tpr, thresholds)
end


"""
Given 2 vectors, `x` and `y`, this function returns the indices that
sort the elements by `x`, with `y` breaking ties. See the example below.
julia> a = [2, 1, 3, 2]
julia> b = [3, 4, 1, 0]
julia> order = sortperm2(a, b)
4-element Array{Int64,1}:
 2
 4
 1
 3
julia> hcat(a[order], b[order]
4×2 Array{Int64,2}:
 1  4
 2  0
 2  3
 3  1
 """
function sortperm2(x, y; rev = false)
    n = length(x)
    no_ties = n == length(Set(x))
    if no_ties
        res = sortperm(x, rev = rev)
    else
        ord1 = sortperm(x, rev = rev)
        x_sorted = x[ord1]
        i = 1
        while i < n

            # println("x_i is $(x_sorted[i]) and x_(i+1) is $(x_sorted[i+1])")
            if x_sorted[i] == x_sorted[i+1]
                if rev && y[ord1][i] < y[ord1][i+1]
                    #println("(1.) Switching $(y[ord1][i]) with $(y[ord1][i+1])")
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                    i = i > 1 ? i - 1 : i
                    continue
                elseif !rev && y[ord1][i] > y[ord1][i+1]
                    #println("(2.) Switching $(y[ord1][i]) with $(y[ord1][i+1])")
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                    i = i > 1 ? i - 1 : i
                    continue
                end
            end
            i += 1
        end
        res = ord1
    end
    return res
end


function _trapsum(y, x)
    d = diff(x)
    res = sum((d .* (y[2:end] + y[1:(end-1)]) ./ 2.0))
    return res
end

function _auc(x, y, reorder = false)
    direction = 1
    if reorder
        order = sortperm2(x, y)
        x, y = x[order], y[order]
    else
        dx = diff(x)
        if any(dx .<= 0)
            if all(dx .<= 0)
                direction = -1
            else
                error("Reordering is not turned on, and the x array is not increasing: $x")
            end
        end
    end
    area = direction * _trapsum(y, x)
    return area
end




"""
    roc_auc_score(y_pred, y_true)
This function returns the area under the curve (AUC) for the receiver operating characteristic 
curve (ROC). This function takes two vectors, `y_true` and `y_pred`. The vector `y_true` is the 
observed `y` in a binary classification problem. And the vector `y_pred` is the real-valued 
prediction for each observation.
"""
function roc_auc_score(y_pred, y_true)
    if length(Set(y_true)) == 1
        warn("Only one class present in y_true.\n
              The AUC is not defined in that case; returning -Inf.")
        res = -Inf
    else
        fpr, tpr, thresholds = _roc_curve(y_true, y_pred)
        res = _auc(fpr, tpr, true)
    end
    # @TODO: add auc
    # res = sum(y_true .== y_pred) / size(y_true, 1)
    return res
end

function accuracy(y_pred, y_true)
    res = sum(y_true .== y_pred) / size(y_true, 1)
    return res;
end


function r2_score(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    mean = sum(y_true) / length(y_true)
    ss_total = sum((y_true .- mean).^2)
    return 1 - ss_res/(ss_total + eps(eltype(y_pred)))
end
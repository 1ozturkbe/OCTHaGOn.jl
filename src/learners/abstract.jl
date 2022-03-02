
EPSILON = 0.5

function classification_evaluation()
    return Pair("Accuracy", accuracy)
end

function regression_evaluation()
    return Pair("R2", r2_score)
end
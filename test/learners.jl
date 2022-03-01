
include("load.jl");

"""
Test SVM 
"""

function test_svm()

    X = rand(100, 50)
    
    #y = 1.0*(rand(100) .> 0.5)
    y = 1.0*(sum(X', dims=[1]).>50)[1,:]#

    lnr = SVM_Classifier(C=5)

    lnr.fit!(lnr, X, y);
    y_hat = lnr.predict(lnr, X)

    accuracy = sum(y .== y_hat)/length(y)
    
    @test accuracy >=1
    println("Accuracy $(accuracy)")
    
end


test_svm()
# OptimalConstraintTree
OptimalConstraintTree is a Julia package that allows for mixed integer 
linear or convex approximations of global optimization problems. 

OCT leans on the [JuMP.jl](https://github.com/jump-dev/JuMP.jl) 
modeling language in its backend, and it develops MI approximations using 
[Interpretable AI](https://www.interpretable.ai/), with a free academic license.
The problems can then be solved by JuMP-compatible solvers, depending on 
the type of approximation. OCT's default solver is [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), 
which is free with an academic license as well. 

Documentation is pending, and the code and testing are under development. 
If you have any burning questions or applications, please create an issue! 

[![codecov.io](http://codecov.io/github/1ozturkbe/OptimalConstraintTree/coverage.svg?branch=master)](http://codecov.io/github/1ozturkbe/OptimalConstraintTree?branch=master)

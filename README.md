# OCTHaGOn

[![codecov](https://codecov.io/gh/1ozturkbe/OCTHaGOn.jl/branch/master/graph/badge.svg?token=3ODJZJN3WT)](https://codecov.io/gh/1ozturkbe/OCTHaGOn.jl)

OCTHaGOn (Optimal Classification Trees with Hyperplanes for Global Optimization) is a [Julia](https://julialang.org/) package that allows for the solution of global optimization problems using mixed-integer (MI) linear and convex approximations. It is an implementation of the methods detailed in [Chapter 2 of this thesis](https://1ozturkbe.github.io/data/ozturk-bozturk-PhD-AeroAstro-2022-thesis.pdf) and submitted to *Operations Research*. OCTHaGOn is licensed under [the MIT License](https://github.com/1ozturkbe/OCTHaGOn.jl/blob/master/LICENSE). 

OCTHaGOn leans on the [JuMP.jl](https://github.com/jump-dev/JuMP.jl) 
modeling language in its backend, and it develops MI approximations using 
[Interpretable AI](https://www.interpretable.ai/), with a free academic license.
The problems can then be solved by JuMP-compatible solvers, depending on 
the type of approximation. OCT's default solver in tests is [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), 
which is free with an academic license as well. 

[Documentation](https://1ozturkbe.github.io/OCTHaGOn.jl/) is available and under development.  
If you have any burning questions or applications, or are having problems with OCTHaGOn, please [create an issue](https://github.com/1ozturkbe/OCTHaGOn.jl/issues)! 
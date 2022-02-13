# Installation

As an overview of the installation steps, the current version of OCTHaGON works with Julia 1.5, with Interpretable AI as its back-end for constraint learning, and CPLEX as its default solver. Please follow the instructions below to get OCTHaGOn working on your machine. 

### Installing Julia

Please find instructions for installing Julia on various platforms [here](https://julialang.org/downloads/). OCTHaGOn is compatible with Julia 1.5, but is frequently tested on Julia 1.5.4, making that the most robust version. However, if you have an existing Julia v1.5.4, we highly recommend you install a clean v1.5.x before proceeding. 

### Installing Interpretable AI

OCTHaGOn requires an installation of Interpretable AI (IAI) for its various machine learning tools. Different builds of IAI are found [here](https://docs.interpretable.ai/stable/download/), corresponding to the version of Julia used. IAI requires a pre-built system image of Julia to replace the existing image (```sys.so``` in Linux and ```sys.dll``` in Windows machines), thus the need for a clean install of Julia v1.5.x. For your chosen v1.5, please replace the system image with the one you downloaded. Then request and deploy an IAI license (free for academics) by following the instructions [here](https://docs.interpretable.ai/stable/installation/). 

### Installing CPLEX

CPLEX is a mixed-integer optimizer that can be found [here](https://www.ibm.com/uk-en/products/ilog-cplex-optimization-studio). It is free to solve optimization problems with up to 1000 variables and constraints, available via [signing up](https://www.ibm.com/account/reg/uk-en/signup?formid=urx-20028), and also available [via a free academic license](https://academic.ibm.com/a2mt/email-auth) for larger problems. 

## Quickest build

Once the above steps are complete, we recommend using the following set of commands as the path of least resistance to getting started. 

Navigate to where you would like to put OCTHaGOn, and call the following commands to instantiate and check all of the dependencies. 

```
git clone https://github.com/1ozturkbe/OCTHaGOn.jl.git
cd OCTHaGOn.jl
julia --project=.
using Pkg
Pkg.instantiate()
```

Call the following to precompile all packages and load OCTHaGOn to your environment:

```julia
include("src/OCTHaGOn.jl")
using .OCTHaGOn
```

Alternatively, you can test your installation of OCTHaGOn by doing the following in a new Julia terminal:

```julia
using Pkg
Pkg.activate("test")
Pkg.instantiate()
include("test/load.jl")
include("test/src.jl")
```

Please see [Basic usage](@ref) for an simple application of OCTHaGOn to a MINLP!
# Introduction to uncertainty management of power systems operations

In this tutorial, we focus on the requirements and deployment of uncertainty management of power system operations. In general, this package is developed using Julia 1.9 + and Python 3.10 + . This package can be developed on Windows, Linux and MacOs with Apple Silicon. 

## The following packages are required:
- Julia 
* JuMP.jl 
* Gurobi.jl
* CPLEX.jl
* SCIP.jl
* Pycall.jl

- Python
* numpy
* scipy

## The following steps are required for joint programming between julia and python

- Pycall
ENV["PYTHON"] should be pointed to the python interpretator, e.g., "/usr/local/bin/python3.10".
- Gurobi 
Installation gurobipy==10.0.3 within python
- 
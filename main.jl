# Import data files in created using Python
using MAT, JuMP, CPLEX, SparseArrays
include("./Solvers/MixedIntegerLinearProgramming.jl")
# Load the .npz file and extract the NumPy arrays
data = matread("problem.mat")

# Access the arrays from the dictionary
problem = data["model_centralized"]
# 
nx = length(problem["obj"])
index_integer = findall(x -> x == "B", problem["vtype"])
index_continuous = findall(x -> x == "B", problem["vtype"])
nc = length(index_continuous)
nb = length(index_integer)
Cc = sparse(1:nc, index_continuous, ones(nc), nc, nx)
Cb = sparse(1:nb, index_integer, ones(nb), nb, nx)
Ac = problem["A"]*Cc'
Ab = problem["A"]*Cb'
b = problem["rhs"]
cc = Cc*problem["obj"]
cb = Cb*problem["obj"]

model = JuMP.direct_model(CPLEX.Optimizer())
@variable(model, xc[1:nc] >= 0)
@variable(model, xb[1:nb] >= 0, Bin)
@constraint(model, Ac * xc + Ab * xb .<= b)

@objective(model, Min, cc' * xc + cb' * xb)
optimize!(model)
xb = value.(xb)
xc = value.(xc)

x_optimal = Cc'*xc + Cb'*xb

# Print the arrays
println(result)
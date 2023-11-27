# Import data files in created using Python
using MAT
include("./Solvers/MixedIntegerLinearProgramming.jl")
# Load the .npz file and extract the NumPy arrays
data = matread("uc_model.mat")

# Access the arrays from the dictionary
problem = data["model"]
result = mixed_integer_linear_programming(vec(problem["obj"]), problem["A"], vec(problem["rhs"]), problem["sense"],vec(problem["lb"]),vec(problem["ub"]), collect(problem["vtype"]), "min")

# Print the arrays
println(result)
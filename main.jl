# Import data files in created using Python
using MAT
include("./Solvers/MixedIntegerLinearProgrammingJUMP.jl")
# Load the .npz file and extract the NumPy arrays
data = matread("problem.mat")

# Access the arrays from the dictionary
problem = data["model_centralized"]
# Split the variables into binary group and continuous group 
result = mixed_integer_linear_programming(problem)
print(result)
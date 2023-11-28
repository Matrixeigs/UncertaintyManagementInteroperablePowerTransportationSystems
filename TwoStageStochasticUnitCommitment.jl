using PowerModels
using MAT
include("./Solvers/MixedIntegerLinearProgrammingJUMP.jl")
# Load the .npz file and extract the NumPy arrays
data = matread("./TestCasesUnitCommitment/problem_gted.mat")

# Access the arrays from the dictionary
problem = data["model_centralized"]


# Split the variables into binary group and continuous group 
result = mixed_integer_linear_programming(problem)
print(result)


print(network_data)

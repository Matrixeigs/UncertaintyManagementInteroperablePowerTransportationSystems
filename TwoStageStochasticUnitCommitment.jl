using PowerModels
using MAT
include("./StochasticOptimization/BenchMark.jl")

# Load the .npz file and extract the NumPy arrays
problem = matread("./TestCasesUnitCommitment/problem_gted.mat")

# Access the data 
model_first_stage = problem["model_first_stage"]
model_second_stage = problem["model_second_stage"]
options = Dict("mipgap"=>1e-3)

# Split the variables into binary group and continuous group 
result = two_stage_so_centralized(model_first_stage, model_second_stage,options)
# print(result)
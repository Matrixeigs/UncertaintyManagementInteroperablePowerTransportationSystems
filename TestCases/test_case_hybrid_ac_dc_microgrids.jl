#=
    Test case for unit commitment within the hybrid AC/DC micorgrids
    where the problem formualtioin is implemented in Python and 
=#

using PyCall

# Update cases pwd 
py_path = pwd() * "/TestCases"
pushfirst!(PyVector(pyimport("sys")."path"), py_path)

problem_formualtion = pyimport("problem_formulation")
data = pyimport("cases_unit_commitment")

mg = data.micro_grid
mg["VOLL"] = 1e9

unit_commitment = problem_formualtion.UnitCommitment(mg)

problem_first_stage, problem_second_stage = unit_commitment.stochastic_optimization(mg)

println(problem_first_stage)        # Output: Name: Julia
println(problem_second_stage) # Output: Greeting: Hello, Julia!
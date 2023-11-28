#=
    Test case for unit commitment within the hybrid AC/DC micorgrids
    where the problem formualtioin is implemented in Python and 
=#

using PyCall
include(pwd()*"/Solvers/MixedIntegerLinearProgrammingJUMP.jl")
# Update cases pwd 
py_path = pwd() * "/TestCases"
pushfirst!(PyVector(pyimport("sys")."path"), py_path)

problem_formualtion = pyimport("problem_formulation")
data = pyimport("cases_unit_commitment")

mg = data.micro_grid
mg["VOLL"] = 1e9

unit_commitment = problem_formualtion.UnitCommitment(mg)

problem_first_stage, problem_second_stage = unit_commitment.stochastic_optimization(mg)

result = mixed_integer_linear_programming(problem_first_stage["c"], problem_first_stage["A"], problem_first_stage["b"], problem_first_stage["Aeq"], problem_first_stage["beq"], problem_first_stage["lb"], problem_first_stage["ub"], problem_first_stage["vtypes"])

println(problem_first_stage)        # Output: Name: Julia
println(problem_second_stage) # Output: Greeting: Hello, Julia!
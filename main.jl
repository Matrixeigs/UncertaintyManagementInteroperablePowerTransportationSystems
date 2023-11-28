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
index_continuous = findall(x -> x == "C", problem["vtype"])
nc = length(index_continuous)
nb = length(index_integer)
Cc = sparse(1:nc, index_continuous, ones(nc), nc, nx)
Cb = sparse(1:nb, index_integer, ones(nb), nb, nx)
Ac = problem["A"]*Cc'
Ab = problem["A"]*Cb'
b = problem["rhs"]
cc = Cc*problem["obj"]
cb = Cb*problem["obj"]
lbc = Cc*problem["lb"]
ubc = Cc*problem["ub"]
lbb = Cb*problem["lb"]
ubb = Cb*problem["ub"]
for i = 1:nc
    if ubc[i] == Inf
        ubc[i] = 1e8
    end
    if lbc[i] == -Inf
        lbc[i] = 1e8
    end
end

# Obtain the equal and inequal constraint indexes
nconstraint = length(problem["sense"])
index_eq = findall(x -> x == "=", problem["sense"])
index_geq = findall(x -> x == ">", problem["sense"])
index_leq = findall(x -> x == "<", problem["sense"])
neq = length(index_eq)
ngeq = length(index_geq)
nleq = length(index_leq)
Ceq = sparse(1:neq, index_eq, ones(neq), neq, nconstraint)
Cgeq = sparse(1:ngeq, index_geq, ones(ngeq), ngeq, nconstraint)
Cleq = sparse(1:nleq, index_leq, ones(nleq), nleq, nconstraint)

model = JuMP.direct_model(CPLEX.Optimizer())
@variable(model, xc[1:nc])
@variable(model, xb[1:nb] >= 0, Bin)
@constraint(model, Ceq * Ac * xc + Ceq * Ab * xb .== Ceq*b)
@constraint(model, Cgeq * Ac * xc + Cgeq * Ab * xb .>= Cgeq * b)
@constraint(model, Cleq * Ac * xc + Cleq * Ab * xb .<= Cleq * b)
@constraint(model, xc .>= lbc)
@constraint(model, xc .<= ubc)
# @constraint(model, xb .>= zeros(nb,1))
# @constraint(model, xb .<= ones(nb,1))
@constraint(model, xb .>= lbb)
@constraint(model, xb .<= ubb)

@constraint(model, Ceq * Ac * xc + Ceq * Ab * xb .== Ceq*b)

@objective(model, Min, cc'*xc + cb' * xb)
optimize!(model)
xb = value.(xb)
xc = value.(xc)

x_optimal = Cc'*xc + Cb'*xb
println(cc'*xc + cb' * xb)

# Print the arrays
println(termination_status(model))

println(objective_value(model))
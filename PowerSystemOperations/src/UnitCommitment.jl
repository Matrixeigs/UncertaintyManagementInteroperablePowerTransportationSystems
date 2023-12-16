#=
    Unit commitment test systems for transmission systems
=#

## Input files
# Parameters of units
Pg_min = [10, 10, 10]
Pg_max = [30, 40, 50]
cg = [20, 15, 10]
bg = [10, 10, 10]

Pd = 80

ng = 3
IG = 1
PG = IG + ng
nx = PG + ng


## Problem formulation using JuMP
using JuMP, Gurobi
model = Model(Gurobi.Optimizer)
set_attribute(model, "TimeLimit", 100)
set_attribute(model, "Presolve", 0)

# Define decision variables
@variable(model, Ig[1:ng], Bin)  # Binary variables Ig
@variable(model, Pg[1:ng] >=0 )  # 

# Define the objective function
@objective(model, Min, cg'*Pg + bg'*Ig)

# Define constraints
@constraint(model, [g=1:ng], Pg[g] <= Ig[g]*Pg_max[g])
@constraint(model, [g=1:ng], Pg[g] >= Ig[g]*Pg_min[g])
@constraint(model, power_balance, sum(Pg) == Pd)

# Solve the optimization problem
optimize!(model)

# Print the optimal solution and objective value
println("Objective value: ", objective_value(model))
println("Optimal solution:")
println("Pg = ", value.(Pg))
println("Ig = ", value.(Ig))


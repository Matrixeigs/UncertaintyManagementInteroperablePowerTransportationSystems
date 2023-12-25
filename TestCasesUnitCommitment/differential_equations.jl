# Using DifferentialEquations.jl to solve ordinary differential equations
using ModelingToolkit, DifferentialEquations, Plots

# Define our state variables: state(t) = initial condition
@variables t x(t)=1 y(t)=1 z(t)=2

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y
       D(y) ~ -γ * y + δ * x * y
       z ~ x + y]

# Bring these pieces together into an ODESystem with independent variable t
@named sys = ODESystem(eqs, t)

# Symbolically Simplify the System
simpsys = structural_simplify(sys)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 10.0)
prob = ODEProblem(simpsys, [], tspan)

# Solve the ODE
sol = solve(prob)

# Plot the solution
p1 = plot(sol, title = "Rabbits vs Wolves")
p2 = plot(sol, idxs = z, title = "Total Animals")

plot(p1, p2, layout = (2, 1))
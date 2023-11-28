#=
    Use the JUMP with high level api to solve mixed integer linear programming problems 
    
    The standard format is:

    \min_{x} c'*x
    s.t. A*x <= b
         lb <= x <= ub
         x_{i} \in J

known prolbem, the GRBaddconstr indexes might not be correct
=#

using JuMP, Gurobi

function mixed_integer_linear_programming(problem::Dict)
    # 1) Reshape the problem
    # 1.1) Split the decision variables into two groups
    nx = length(problem["obj"])
    index_integer = findall(x -> x == "B", problem["vtype"])
    index_continuous = findall(x -> x == "C", problem["vtype"])
    nc = length(index_continuous)
    nb = length(index_integer)
    Cc = sparse(1:nc, index_continuous, ones(nc), nc, nx)
    Cb = sparse(1:nb, index_integer, ones(nb), nb, nx)
    # 1.2) Split the constraints into three groups
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
    # 1.3) Obtain the equal and inequal constraint indexes
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
    # 2) problem formualtion
    model = JuMP.direct_model(Gurobi.Optimizer())
    # 2.1) add variables
    @variable(model, xc[1:nc])
    @variable(model, xb[1:nb] >= 0, Bin)
    @constraint(model, xc .>= lbc)
    @constraint(model, xc .<= ubc)
    @constraint(model, xb .>= lbb)
    @constraint(model, xb .<= ubb)
    # 2.2) add constraints
    @constraint(model, Ceq * Ac * xc + Ceq * Ab * xb .== Ceq*b)
    @constraint(model, Cgeq * Ac * xc + Cgeq * Ab * xb .>= Cgeq * b)
    @constraint(model, Cleq * Ac * xc + Cleq * Ab * xb .<= Cleq * b)
    # 2.3) define the objective
    @objective(model, Min, cc'*xc + cb' * xb)
    # 3) solve the model
    set_optimizer_attribute(model, "OutputFlag", 0)
    optimize!(model)
    xb = value.(xb)
    xc = value.(xc)
    # 4) recover the solution
    x_optimal = Cc'*xc + Cb'*xb
    sol = Dict("x" => x_optimal, "objval" => objective_value(model), "status" => termination_status(model))
    return sol
end

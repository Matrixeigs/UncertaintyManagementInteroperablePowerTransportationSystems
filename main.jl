using JuMP, GLPK

# Define a simple MILP problem for demonstration purposes
function define_milp()
    model = Model(GLPK.Optimizer)
    @variable(model, x, Int)
    @variable(model, y >= 0)
    @objective(model, Max, x + 2y)
    @constraint(model, 1x + 1y <= 10)
    return model
end

# Branch and Bound algorithm
function branch_and_bound(model, epsilon=1e-5)
    # Solve the LP relaxation of the problem
    set_optimizer_attribute(model, MOI.Silent(), true)
    optimize!(model)
    
    # Check if the solution is already integer
    if is_integer_solution(model)
        return value.(all_variables(model)), objective_value(model)
    end
    
    # Branching
    # Here we just pick the first non-integer variable to branch on
    x_val, x_var = find_branching_variable(model)
    lower_bound, upper_bound = floor(x_val), ceil(x_val)
    
    # Create and solve the "left" branch
    left_model = copy_model(model)
    @constraint(left_model, x_var <= lower_bound)
    left_solution, left_obj = branch_and_bound(left_model)
    
    # Create and solve the "right" branch
    right_model = copy_model(model)
    @constraint(right_model, x_var >= upper_bound)
    right_solution, right_obj = branch_and_bound(right_model)
    
    # Choose the better branch
    if left_obj > right_obj
        return left_solution, left_obj
    else
        return right_solution, right_obj
    end
end

# Utility function to check if current solution is integer
function is_integer_solution(model)
    for var in all_variables(model)
        if !isinteger(value(var))
            return false
        end
    end
    return true
end

# Find a variable to branch on
function find_branching_variable(model)
    for var in all_variables(model)
        val = value(var)
        if !isinteger(val)
            return val, var
        end
    end
end

# Copy model utility (to keep the original model intact while branching)
function copy_model(model)
    return deepcopy(model)
end

# Main function to solve the MILP
function solve_milp()
    milp_model = define_milp()
    solution, objective = branch_and_bound(milp_model)
    println("Optimal solution: ", solution)
    println("Optimal objective: ", objective)
end

# Call the main function
solve_milp()

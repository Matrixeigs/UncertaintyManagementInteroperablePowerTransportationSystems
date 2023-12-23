# Write a function that takes in a vector of cars and returns a vector of cars
using Agents

mutable struct Vehicle <: AbstractAgent
    id::Int
    pos::Tuple{Int,Int}
    speed::Float64
    destination::Tuple{Int,Int}
    # Add other properties like direction, etc.
end


function initialize_model(num_vehicles::Int, grid_size::Tuple{Int,Int})
    space = GridSpace(grid_size)
    model = ABM(Vehicle, space)

    for id in 1:num_vehicles
        pos = (rand(1:grid_size[1]), rand(1:grid_size[2]))
        speed = rand()  # Random speed
        destination = (rand(1:grid_size[1]), rand(1:grid_size[2]))
        add_agent!(pos, model, speed, destination)
    end

    return model
end

model = initialize_model(100, (10, 10))  # 100 vehicles in a 10x10 grid

function move_vehicle!(vehicle::Vehicle, model)
    # Determine the range of possible moves
    # For simplicity, let's allow the vehicle to move up, down, left, or right by 1 grid space
    possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Randomly select a move from the possible moves
    move = rand(possible_moves)
    
    # Calculate the new position
    new_pos = (vehicle.pos[1] + move[1], vehicle.pos[2] + move[2])
    
    # Ensure the new position is within the grid bounds
    new_pos = (
        max(1, min(new_pos[1], size(model.space)[1])),  # x-coordinate
        max(1, min(new_pos[2], size(model.space)[2]))   # y-coordinate
    )
    
    # Move the agent to the new position
    move_agent!(vehicle, new_pos, model)
end

using Plots

function visualize_traffic(model)
    # Extract the positions of all vehicles
    positions = [(agent.pos[1], agent.pos[2]) for agent in allagents(model)]
    
    # Create a scatter plot of the positions
    plot = scatter([pos[1] for pos in positions], 
                   [pos[2] for pos in positions], 
                   markersize = 5,
                   color = :blue,
                   xlims = (0, size(model.space)[1]),
                   ylims = (0, size(model.space)[2]),
                   xlabel = "X",
                   ylabel = "Y",
                   title = "Traffic Simulation",
                   legend = false)

    # Display the plot
    display(plot)
end



for i in 1:100  # Run for 100 steps
    step!(model, move_vehicle!, 1)
    visualize_traffic(model)
    sleep(0.1)  # Pause briefly between steps to create an animation effect
end
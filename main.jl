# Import data files in created using Python
using NPZ

# Load the .npz file and extract the NumPy arrays
data = npzread("problem_vrp.npz")

# Access the arrays from the dictionary
array1 = data["cobj"]
array2 = data["lb"]

# Print the arrays
println(array1)
println(array2)
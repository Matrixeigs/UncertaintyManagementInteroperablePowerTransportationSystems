# Write a surrograte model for the frequency stability problem
using QuasiMonteCarlo, MAT, Surrogates, SurrogatesFlux, Flux
data = matread("data_grid.mat")
sample_points = data["x"];
df_nadir = data["y"];

# Transpose sample_points
sample_points = transpose(sample_points)
# Obtain the lower and upper bounds
lower_bound = minimum(sample_points, dims = 2)
upper_bound = maximum(sample_points, dims = 2)

# Build a surrogate model for the frequency stability criteria
n_samples = size(sample_points, 2)
n_features = size(sample_points, 1)
n_targets = size(df_nadir, 1)
# Convert sample_points matrix to a tuple of vectors for Flux with data precision
sample_points = tuple([Float16.(sample_points[:, i]) for i in 1:n_samples]...)

model1 = Chain(
  Dense(n_features, 5, σ),
  Dense(5, 2, σ),
  Dense(2, 1)
)
neural = NeuralSurrogate(sample_points, df_nadir, lower_bound, upper_bound, model = model1, n_echos = 10)
# we need a specific function to approximate
surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, neural, SobolSample(), maxiters=20, num_new_samples=10)
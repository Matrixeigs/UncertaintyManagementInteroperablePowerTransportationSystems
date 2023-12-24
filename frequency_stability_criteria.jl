# A function to approximate the frequency stability criteria for a given power systems
# Author: Tianyang Zhao
# Date: 2023-12-23
# License: MIT License
# Version: 0.1
# Reference:
# [1] Anderson, Philip M., and Mahmood Mirheydar. "A low-order system frequency response model." IEEE transactions on power systems 5.3 (1990): 720-729.


# Import packages
using QuasiMonteCarlo, MAT, Surrogates, SurrogatesFlux, Flux, Plots

# Define a function to generate random numbers
H_min = 2.6344;
H_max = 4.6344;
R_min = 9.50;
R_max = 19.0;
D_min = 4.9405;
D_max = 5.4405;
delta_P_min = 0.05;
delta_P_max = 0.20;


lb = [H_min, R_min, D_min, delta_P_min];
ub = [H_max, R_max, D_max, delta_P_max];
n_features = length(lb);

# Generate n sample points using QuasiMonteCarlo package
n_samples = 1000;
# Set the QuasiMonteCarlo precision to Float32
sample_points = sample(n_samples, lb, ub, SobolSample());

# sample_points = QuasiMonteCarlo.sample(n_samples, lb, ub, SobolSample());
# sample_points = QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())
# boxplot sample_points
# Define two global variables D and T_g
global T_g = 8.0;
global M = 2*H_max;
global F_gsys = 5.70;
global R_gsys = 19.0;


# Define a function to calculate the frequency stability criteria
function frequency_response(H::Float64, R::Float64, D::Float64, delta_P::Float64)
    # Define the parameters
    H = H;
    R = R;
    D = D;
    delta_P = delta_P;
    # Calculate the frequency response
    w_n = sqrt((D + R)/(M * T_g));
    et = (M + T_g * (D + F_gsys))/(2 * sqrt(M * T_g * (D + R)));

    
    if et > 1
        w_c = w_n * sqrt(Complex(1 - et^2));
        theta = asin(sqrt(Complex(1 - et^2)));
    else
        w_c = w_n * sqrt(1 - et^2);
        theta = asin(sqrt(1 - et^2));
    end

    t_nadir = 1 / w_c * atan(w_c * T_g / (et * w_n * T_g - 1));
    if typeof(t_nadir) == Complex{Float32}
        t_nadir = real(t_nadir)
    end

    f_nadir = delta_P / (D + R_gsys) * (1 + sqrt(T_g * (R_gsys - F_gsys) / M) * exp(-et * w_n * t_nadir));
    if typeof(f_nadir) == Complex{Float32}
        f_nadir = real(f_nadir)
    end
    return f_nadir
end 

function frequency_response(x::NTuple{4, Float64})
    # Define the parameters
    H = x[1];
    R = x[2];
    D = x[3];
    delta_P = x[4];
    # Calculate the frequency response
    w_n = sqrt((D + R)/(M * T_g));
    et = (M + T_g * (D + F_gsys))/(2 * sqrt(M * T_g * (D + R)));

    if et > 1
        w_c = w_n * sqrt(Complex(1 - et^2));
        theta = asin(sqrt(Complex(1 - et^2)));
    else
        w_c = w_n * sqrt(1 - et^2);
        theta = asin(sqrt(1 - et^2));
    end

    t_nadir = 1 / w_c * atan(w_c * T_g / (et * w_n * T_g - 1));
    # If the type of t_nadir is Complex, then convert it to Float64
    if typeof(t_nadir) == Complex{Float64}
        t_nadir = real(t_nadir)
    end

    f_nadir = delta_P / (D + R_gsys) * (1 + sqrt(T_g * (R_gsys - F_gsys) / M) * exp(-et * w_n * t_nadir));
    if typeof(f_nadir) == Complex{Float64}
        f_nadir = real(f_nadir)
    end
    return f_nadir
end


        # Calculate the frequency stability criteria for each sample point
# t_nadir = zeros(n_samples);
# df_nadir = zeros(n_samples);
# for i in 1:n_samples
#     t_nadir[i], df_nadir[i] = frequency_response(sample_points[1, i], sample_points[2, i], sample_points[3, i], sample_points[4, i]);
# end
df_nadir = frequency_response.(sample_points);
# Plot the frequency response with respect to the sample points
# show the plot
# plot!(sample_points[1, :], df_nadir, seriestype = :scatter, label = "df_nadir")


# Build a surrogate model for the frequency stability criteria
# matwrite("data_grid.mat", Dict("x" => sample_points, "y" => df_nadir))

# Train a surrogate model with ReLU activation function
model = Chain(
    Dense(n_features, 5, σ),
    Dense(5, 2, σ),
    Dense(2, 1)
)

# sample_points = tuple([Float64.(sample_points[:, i]) for i in 1:n_samples]...);
neural = NeuralSurrogate(sample_points, df_nadir, lb, ub, model = model, n_echos = 5)
# linear_mode = LinearSurrogate(sample_points, df_nadir, lb, ub)
# optimize the surrogate model with Expected Improvement acquisition function
# Assess the performance of the surrogate model
error = zeros(n_samples);
for i in 1:n_samples
    error[i] = abs(frequency_response(sample_points[i]) - neural(sample_points[i]));
end
# Analyse the error 
plot!(df_nadir, error, seriestype = :scatter, label = "error")

surrogate_optimize(frequency_response, SRBF(), lb, ub, neural, SobolSample(), maxiters=20, num_new_samples=10)
# Export the surrogate model
matwrite("neural.mat", Dict("neural" => neural))
# A function to approximate the frequency stability criteria for a given power systems
# Author: Tianyang Zhao
# Date: 2023-12-23
# License: MIT License
# Version: 0.1

# Import packages
using QuasiMonteCarlo, Plots

# Define a function to generate random numbers
H_min = 2;
H_max = 4;
R_min = 0.025;
R_max = 0.05;
F_min = 0.3;
F_max = 0.3;
delta_P_min = 0.05;
delta_P_max = 0.25;


lb = [H_min, R_min, F_min, delta_P_min];
ub = [H_max, R_max, F_max, delta_P_max];
nx = length(lb);

# Generate n sample points using QuasiMonteCarlo package
n = 1000;

sample_points = QuasiMonteCarlo.sample(n, lb, ub, SobolSample())
# sample_points = QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())

# Define a function to calculate the frequency stability criteria
function frequency_response(H::Float64, R::Float64, F::Float64, delta_P::Float64)
    # Define the parameters
    H = H;
    R = R;
    F = F;
    delta_P = delta_P;
    # Calculate the frequency response
    # Define the parameters
    D = 1;
    T_g = 8;
    # Calculate the frequency response
    w_n = sqrt((D * R + 1) / (2 * H * R * T_g));
    et = (D * R * T_g + 2 * H * R + F * T_g)/2/(D * R + 1) * w_n;
    w_r = w_n * sqrt(1 - et^2)
    alpha = sqrt((1 - 2 * T_g * et * w_n + T_g^2 * w_n^2)/(1 - et^2));
    theta = atan(w_r * T_g / (1 - et * w_n * T_g)) - atan(-sqrt(1 - et^2)/et);
    t_nadir = 1/w_r * atan(w_r * T_g / (et * w_r * T_g -1));
    df_nadir = R * delta_P / (D * R + 1) *  (1 + alpha * exp(- et * w_n * t_nadir) * sin(w_r * t_nadir + theta));
    return t_nadir, df_nadir
end 

# Calculate the frequency stability criteria for each sample point
t_nadir = zeros(n);
df_nadir = zeros(n);
for i in 1:n
    t_nadir[i], df_nadir[i] = frequency_response(sample_points[1,i], sample_points[2,i], sample_points[3,i], sample_points[4,i]);
end

# Build a surrogate model for the frequency stability criteria



# Plot the frequency stability criteria
plot(t_nadir, df_nadir, seriestype = :scatter, markersize = 2, xlabel = "t_nadir", ylabel = "df_nadir", legend = false)

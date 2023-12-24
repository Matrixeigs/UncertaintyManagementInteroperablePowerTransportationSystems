# A function to approximate the frequency stability criteria for a given power systems
# Author: Tianyang Zhao
# Date: 2023-12-23
# License: MIT License
# Version: 0.1
# Reference:
# [1] A Low-Order System Frequency Response Model


# Import packages
using QuasiMonteCarlo, MAT

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
n = 1000;

sample_points = QuasiMonteCarlo.sample(n, lb, ub, SobolSample());
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
    if typeof(t_nadir) == Complex{Float64}
        t_nadir = real(t_nadir)
    end

    f_nadir = delta_P / (D + R_gsys) * (1 + sqrt(T_g * (R_gsys - F_gsys) / M) * exp(-et * w_n * t_nadir));
    if typeof(f_nadir) == Complex{Float64}
        f_nadir = real(f_nadir)
    end
    return t_nadir, f_nadir
end 

        # Calculate the frequency stability criteria for each sample point
t_nadir = zeros(n);
df_nadir = zeros(n);
for i in 1:n
    t_nadir[i], df_nadir[i] = frequency_response(sample_points[1, i], sample_points[2, i], sample_points[3, i], sample_points[4, i]);
end

# Build a surrogate model for the frequency stability criteria
matwrite("data_grid.mat", Dict("x" => sample_points, "y" => df_nadir, "t_nadir" => t_nadir))

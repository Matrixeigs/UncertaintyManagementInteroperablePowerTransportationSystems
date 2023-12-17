using Distributions
using StatsBase
using Plots

# Define the multivariate probability density function (replace this with your PDF)
pdf = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])  # Example: Multivariate normal distribution with mean [0, 0] and covariance matrix

# Function to estimate confidence level using Monte Carlo simulation
function estimate_confidence_level_multivariate(pdf::Distribution, n::Int, α::Float64)
    samples = rand(pdf, n)  # Generate random samples from the multivariate PDF
    
    # Perform calculations using the samples to estimate confidence level (adapt this to your specific case)
    # For instance, you might compute confidence ellipsoids, check if a certain point is within the interval, etc.
    # Here, let's assume a simple case of checking if the mean is within a confidence interval

    # Calculate mean of samples
    mean_samples = mean(samples, dims=1)
    # display(heatmap(mean_samples, xlabel="Index", ylabel="Values", title="Mean value"))
    
    # Calculate confidence interval based on mean and standard error
    confidence_interval = (quantile(Normal(), α / 2), quantile(Normal(), 1 - α / 2))
    
    # Check if mean is within the confidence interval
    is_in_interval = confidence_interval[1] <= mean_samples <= confidence_interval[2]
    confidence_level = is_in_interval ? 1 - α : α
    
    return confidence_level
end

# Number of samples and confidence level (alpha)
n_samples = 1000
confidence_alpha = 0.05  # 95% confidence level

# Estimate confidence level using Monte Carlo simulation
confidence_level = estimate_confidence_level_multivariate(pdf, n_samples, confidence_alpha)

println("Estimated confidence level: $(confidence_level * 100)%")
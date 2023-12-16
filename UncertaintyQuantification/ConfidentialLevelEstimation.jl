using Distributions
using StatsBase

# Define the probability density function (replace this with your PDF)
pdf = Normal(0, 1)  # Example: Normal distribution with mean 0 and standard deviation 1

# Function to estimate confidence level using bootstrapping
function estimate_confidence_level(pdf::Distribution, x::Real, n::Int, α::Float64)
    samples = rand(pdf, n)  # Generate random samples from the PDF
    confidence_interval = percentile(range(0, 100, length = n), α / 2), percentile(range(0, 100, length = n), 1 - α / 2)
    is_in_interval = confidence_interval[1] <= x <= confidence_interval[2]
    confidence_level = is_in_interval ? 1 - α : α
    return confidence_level
end

# Point at which confidence level is to be estimated
x = 0.5

# Number of samples and confidence level (alpha)
n_samples = 1000
confidence_alpha = 0.05  # 95% confidence level

# Estimate confidence level using bootstrapping
confidence_level = estimate_confidence_level(pdf, x, n_samples, confidence_alpha)

println("Estimated confidence level at $x: $(confidence_level * 100)%")

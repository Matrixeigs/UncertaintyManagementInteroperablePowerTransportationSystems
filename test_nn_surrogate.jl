Base.compilecache(Base.identify_package("GR"))
using Plots
using Surrogates
using Flux
using SurrogatesFlux

function schaffer(x)
    x1=x[1]
    x2=x[2]
    fact1 = x1 ^2;
    fact2 = x2 ^2;
    y = fact1 + fact2;
end

n_samples = 60
lower_bound = [0.0, 0.0]
upper_bound = [8.0, 8.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = schaffer.(xys);

model1 = Chain(
  Dense(2, 5, σ),
  Dense(5,2,σ),
  Dense(2, 1)
)
neural = NeuralSurrogate(xys, zs, lower_bound, upper_bound, model = model1, n_echos = 10)

surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, neural, SobolSample(), maxiters=20, num_new_samples=10)

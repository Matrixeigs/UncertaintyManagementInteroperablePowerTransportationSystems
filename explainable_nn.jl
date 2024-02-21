using Flux

model = Chain(
  Dense(784, 128, relu),
  Dense(128, 64, relu),
  Dense(64, 10),
  softmax
)

using CairoMakie

# Assuming your model's first layer is a Dense layer
weights = Flux.params(model)[1]
figure = Figure()
heatmap(figure[1, 1], weights)
figure

using Pluto
Pluto.run()
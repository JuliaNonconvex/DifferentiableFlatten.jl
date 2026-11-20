# Differentiation

```julia
using DifferentiableFlatten, Zygote

x = (a = [1.0, 2.0], b = 3.0)
v, unflatten = DifferentiableFlatten.flatten(x)
g = Zygote.gradient(v -> sum(abs2, unflatten(v)), v)  # gradient wrt. v
```

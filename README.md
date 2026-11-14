# DifferentiableFlatten

[![Actions Status](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/workflows/CI/badge.svg)](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/actions)
[![codecov](https://codecov.io/gh/JuliaNonconvex/DifferentiableFlatten.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/DifferentiableFlatten.jl)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/DifferentiableFlatten.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/DifferentiableFlatten.jl/dev)

A `flatten` function which flattens arbitrary nested data structures (arrays,
tuples, named tuples, dictionaries, structs, and combinations thereof) to
real vectors, and returns an unflattening function to reconstruct the
original data structure. `flatten` and its unflattening function are
differentiable through ChainRules `rrule`s, so differentiation with Zygote
and other ChainRules-based systems works through the flattening. This was
originally part of [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).
It is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex)
ecosystem behind [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl).

The package is unexported on purpose: all names are accessed qualified, e.g.
`DifferentiableFlatten.flatten`.

## Usage

```julia
using DifferentiableFlatten

x = (a = [1.0, 2.0], b = (c = 3.0, d = randn(2, 2)))
v, unflatten = DifferentiableFlatten.flatten(x)
# v is a Vector{Float64}; unflatten(v) ≈ x

y = unflatten(v)   # a NamedTuple with the same structure and values as x
```

Flattening works on plain structs, tuples, vectors of mixed types, and
dictionaries (dictionaries are flattened in their key order, which can be
made deterministic with `OrderedDict`).

For custom structs whose reconstruction needs special handling (for example
a parametric type whose parameters should be fixed), the `@constructor`
macro generates the specialized `flatten` and `zygote_flatten` methods:

```julia
using DifferentiableFlatten
import DifferentiableFlatten: @constructor

struct MyParametric{T}
    data::Vector{T}
end

# Reconstruct MyParametric{Float64} from any MyParametric{T}:
DifferentiableFlatten.@constructor(MyParametric, MyParametric{Float64})

x = MyParametric{Float32}([1, 2, 3])
v, unflatten = DifferentiableFlatten.flatten(x)
unflatten(v)  # MyParametric{Float64}([1.0, 2.0, 3.0])
```

See the [documentation](https://julianonconvex.github.io/DifferentiableFlatten.jl/stable)
for the full API reference.

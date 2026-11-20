# Usage

```julia
using DifferentiableFlatten

x = (a = [1.0, 2.0], b = (c = 3.0, d = randn(2, 2)))
v, unflatten = DifferentiableFlatten.flatten(x)
# v is a Vector{Float64}; unflatten(v) ≈ x

y = unflatten(v)   # a NamedTuple with the same structure and values as x
```

Flattening also works on plain structs, tuples, vectors of mixed types, and
dictionaries (dictionaries are flattened in their key order, which can be
made deterministic with `OrderedDict`).

For custom structs whose reconstruction needs special handling (for example
a parametric type whose parameters should be fixed), the
[`@constructor`](@ref DifferentiableFlatten.@constructor) macro generates
the specialized `flatten` and `zygote_flatten` methods:

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

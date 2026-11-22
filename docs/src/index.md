# DifferentiableFlatten

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/DifferentiableFlatten.jl/main/docs/src/assets/logo.png)

A [`flatten`](@ref DifferentiableFlatten.flatten) function which flattens
arbitrary nested data structures
(arrays, tuples, named tuples, dictionaries, structs, and combinations
thereof) into real vectors, and returns an unflattening function to
reconstruct the original data structure. `flatten` and its unflattening
function are differentiable: ChainRules `rrule`s are provided so that
differentiation with Zygote and other ChainRules-based systems works through
the flattening, which is what makes it the backbone of the structured
variable support in [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).
This was originally part of NonconvexCore.jl.

The package is unexported on purpose: all names are accessed qualified, e.g.
`DifferentiableFlatten.flatten`.


See [Usage](usage.md) for the flattening API and
[Differentiation](differentiation.md) for the ChainRules support and
[Docstrings](docstrings.md) for the full API reference.

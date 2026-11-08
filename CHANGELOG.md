# Changelog

All notable changes to `DifferentiableFlatten.jl` are recorded here.
The format follows the recommendation from JuliaRegistrator:
a brief summary, then categorized lists of breaking changes,
new features, bug fixes, and maintenance items.

## v0.2.1 (2026-11-08)

### Bug fixes
- The pullback of `Unflatten` for dictionaries now returns the flat cotangent
  in the forward key order. It previously used the key order of the merged
  cotangent dictionary, which is sorted, so reverse-mode gradients through
  dictionaries with unsorted insertion order were permuted. ([#11](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/pull/11))

## v0.2.0 (2026-08-11)

### Breaking changes
- Minimum Julia version raised from `1` to `1.10`.
  Consumers of this package must compile and test on Julia 1.10+
  ([#5](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/pull/5)).

### Maintenance
- Standardized `[compat]` entries; stdlib bounds normalized to `"1"`.
  ([#5](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/pull/5)).

## v0.1.2 (2026-08-08)

### Bug fixes
- Handle AbstractThunk co-tangents for Zygote 0.7 compatibility
  ([#4](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl/pull/4)).

## v0.1.0 (2022-10-17)

### Initial release
- Zygote-differentiable flatten and unflatten helpers, split out of
  NonconvexCore.jl.

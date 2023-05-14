module DifferentiableFlattenJuMPExt
if isdefined(Base, :get_extension)
    import DifferentiableFlatten
    using DifferentiableFlatten: flatten,Array_from_vec,zygote_flatten,from_vec
    import JuMP
else
    using ..DifferentiableFlatten
    using ..JuMP
end


function DifferentiableFlatten.flatten(x::JuMP.Containers.DenseAxisArray)
    x_vec, from_vec = flatten(vec(identity.(x.data)))
    Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x)), axes(x)...)
    return identity.(x_vec), Array_from_vec
end

function DifferentiableFlatten.zygote_flatten(x1::JuMP.Containers.DenseAxisArray, x2::NamedTuple)
    x_vec, from_vec = zygote_flatten(vec(identity.(x1.data)), vec(identity.(x2.data)))
    Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
    return identity.(x_vec), Array_from_vec
end

function DifferentiableFlatten.zygote_flatten(x1::JuMP.Containers.DenseAxisArray, x2::JuMP.Containers.DenseAxisArray)
    x_vec, from_vec = zygote_flatten(vec(identity.(x1.data)), vec(identity.(x2.data)))
    Array_from_vec(x_vec) = JuMP.Containers.DenseAxisArray(reshape(from_vec(x_vec), size(x2)), axes(x2)...)
    return identity.(x_vec), Array_from_vec
end

end
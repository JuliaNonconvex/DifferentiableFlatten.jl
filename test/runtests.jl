using DifferentiableFlatten: flatten, zygote_flatten
using OrderedCollections, JuMP, Zygote, SparseArrays, LinearAlgebra, Test

struct SS
    a
    b
end

@testset "DifferentiableFlatten.jl" begin    
    xs = [
        1.0,
        [1.0],
        [1.0, 2.0],
        [1.0, Float64[1.0, 2.0]],
        [1.0, (1.0, 2.0)],
        [1.0, OrderedDict(1 => Float64[1.0, 2.0])],
        [[1.0], OrderedDict(1 => Float64[1.0, 2.0])],
        [(1.0,), [1.0,], OrderedDict(1 => Float64[1.0, 2.0])],
        [1.0 1.0; 1.0 1.0],
        rand(2, 2, 2),
        [Float64[1.0, 2.0], Float64[3.0, 4.0]],
        OrderedDict(1 => 1.0),
        OrderedDict(1 => Float64[1.0]),
        OrderedDict(1 => 1.0, 2 => Float64[2.0]),
        OrderedDict(1 => 1.0, 2 => Float64[2.0], 3 => [Float64[1.0, 2.0], Float64[3.0, 4.0]]),
        JuMP.Containers.DenseAxisArray(reshape(Float64[1.0, 1.0], (2,)), 1),
        (1.0,),
        (1.0, 2.0),
        (1.0, (1.0, 2.0)),
        (1.0, Float64[1.0, 2.0]),
        (1.0, OrderedDict(1 => Float64[1.0, 2.0])),
        ([1.0], OrderedDict(1 => Float64[1.0, 2.0])),
        ((1.0,), [1.0,], OrderedDict(1 => Float64[1.0, 2.0])),
        (a = 1.0,),
        (a = 1.0, b = 2.0),
        (a = 1.0, b = (1.0, 2.0)),
        (a = 1.0, b = Float64[1.0, 2.0]),
        (a = 1.0, b = OrderedDict(1 => Float64[1.0, 2.0])),
        (a = [1.0], b = OrderedDict(1 => Float64[1.0, 2.0])),
        (a = (1.0,), b = [1.0,], c = OrderedDict(1 => Float64[1.0, 2.0])),
        sparsevec(Float64[1.0, 2.0], [1, 3], 10),
        sparse([1, 2, 2, 3], [2, 3, 1, 4], Float64[1.0, 2.0, 3.0, 4.0], 10, 10),
        SS(1.0, 2.0),
        [SS(1.0, 2.0), 1.0],
    ]
    for x in xs
        @show x
        xvec, unflatten = flatten(x)
        @test x == unflatten(xvec)
        J = Zygote.jacobian(xvec) do x
            unflatten(x)
            flatten(x)[1]
        end[1]
        @test logabsdet(J) == (0.0, 1.0)

        xvec, unflatten = zygote_flatten(x, x)
        @test x == unflatten(xvec)
        J = Zygote.jacobian(xvec) do x
            unflatten(x)
            zygote_flatten(x, x)[1]
        end[1]
        @test logabsdet(J) == (0.0, 1.0)
    end
end

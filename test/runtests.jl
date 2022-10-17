using DifferentiableFlatten: flatten, zygote_flatten, maybeflatten
using DifferentiableFlatten: DifferentiableFlatten, @constructor
using OrderedCollections, JuMP, Zygote, SparseArrays, LinearAlgebra, Test
using ChainRulesCore

struct SS
    a
    b
end

struct MyStruct{T, T1, T2}
    a::T1
    b::T2
end
MyStruct(a, b) = MyStruct{typeof(a), typeof(a), typeof(b)}(a, b)
@constructor MyStruct MyStruct

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
        MyStruct(1.0, 1.0),
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
        if x isa Real
            @test maybeflatten(x) == x
        else
            xvec, unflatten = maybeflatten(x)
            @test x == unflatten(xvec)
        end
        @show DifferentiableFlatten._zero(x)
        @test all(==(0), flatten(DifferentiableFlatten._zero(x))[1])
    end

    xvec, unflatten = zygote_flatten(SS(1.0, 2.0), Tangent{SS}(a = 1.0, b = 2.0))
    @test unflatten(xvec) isa NamedTuple

    xvec, unflatten = zygote_flatten(SS(1.0, 2.0), (a = 1.0, b = 2.0))
    @test unflatten(xvec) isa NamedTuple
    @test DifferentiableFlatten._length(nothing) == 0

    @test DifferentiableFlatten._merge(
        OrderedDict(:a => 1.0),
        OrderedDict(:b => 2.0),
    ) == OrderedDict(:a => 0.0, :b => 2.0)
    @test DifferentiableFlatten._merge(1, SS(1.0, 2.0)) == SS(1.0, 2.0)
    x = OrderedDict(:a => 1.0)
    @test DifferentiableFlatten._merge(
        (1.0,),
        Tangent{NamedTuple{(:b,), Tuple{Float64}}}(b = 1.0),
    ) == (ZeroTangent(),)

    @test flatten(nothing)[1] == Float64[]
    @test flatten(NoTangent())[1] == Float64[]
    @test flatten(ZeroTangent())[1] == Float64[]
    @test flatten(())[1] == Float64[]

    @test zygote_flatten(1.0, nothing)[1] == [0.0]
    @test zygote_flatten(1.0, NoTangent())[1] == [0.0]
    @test zygote_flatten(1.0, ZeroTangent())[1] == [0.0]
    @test zygote_flatten(1.0, ())[1] == Float64[]

    x = JuMP.Containers.DenseAxisArray(reshape(Float64[1.0, 1.0], (2,)), 1)
    @test zygote_flatten(x, (data = [1.0, 1.0],))[1] == [1.0, 1.0]
end

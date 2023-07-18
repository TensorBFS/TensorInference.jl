using Test, TensorInference, Documenter, Pkg

@testset "MAR" begin
    include("mar.jl")
end

@testset "MAP" begin
    include("map.jl")
end

@testset "MMAP" begin
    include("mmap.jl")
end

@testset "PR" begin
    include("pr.jl")
end

@testset "sampling" begin
    include("sampling.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual = false)

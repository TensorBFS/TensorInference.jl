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

@testset "cspmodels" begin
    include("cspmodels.jl")
end

@testset "utils" begin
    include("utils.jl")
end

@testset "belief propagation" begin
    include("belief.jl")
    include("loop_series.jl")
end

@testset "fileio" begin
    include("fileio.jl")
end

@testset "RescaledArray" begin
    include("RescaledArray.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual = false)

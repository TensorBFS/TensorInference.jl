using Test, TensorInference, Documenter

@testset "inference" begin
  include("inference.jl")
end

@testset "MAP" begin
  include("maxprob.jl")
end

@testset "MMAP" begin
  include("mmap.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual=false)

using Test, TensorInference

@testset "inference" begin
  include("inference.jl")
end

@testset "MAP" begin
  include("maxprob.jl")
end

@testset "MMAP" begin
  include("mmap.jl")
end
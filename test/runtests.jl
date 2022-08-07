using Test, TensorInference

@testset "inference" begin
  include("inference.jl")
end

@testset "MAP" begin
  include("maxprob.jl")
end
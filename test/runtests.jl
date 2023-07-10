using Test, TensorInference, Documenter, Pkg, Artifacts

import Pkg;
Pkg.ensure_artifact_installed("uai2014", joinpath(@__DIR__, "Artifacts.toml"));

include("utils.jl")

@testset "MAR" begin
    include("mar.jl")
end

@testset "MAP" begin
    include("map.jl")
end

@testset "MMAP" begin
    include("mmap.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual = false)

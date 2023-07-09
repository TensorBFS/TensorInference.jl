using Test, TensorInference, Documenter, Pkg, Artifacts

import Pkg;
Pkg.ensure_artifact_installed("uai2014", "Artifacts.toml");

include("utils.jl")

@testset "MAR" begin
    include("mar.jl")
end

@testset "MPE" begin
    include("mpe.jl")
end

@testset "MMAP" begin
    include("mmap.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual = false)

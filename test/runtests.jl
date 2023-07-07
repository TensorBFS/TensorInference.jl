using Test, TensorInference, Documenter, Pkg

artifacts_toml = "Artifacts.toml"
artifacts = Pkg.Artifacts.select_downloadable_artifacts(artifacts_toml)
for name in keys(artifacts)
    Pkg.Artifacts.ensure_artifact_installed(name, artifacts[name], artifacts_toml)
end

@testset "inference" begin
    include("inference.jl")
end

@testset "MAP" begin
    include("maxprob.jl")
end

@testset "MMAP" begin
    include("mmap.jl")
end
@testset "MMAP" begin
    include("sampling.jl")
end

using CUDA
if CUDA.functional()
    include("cuda.jl")
end

Documenter.doctest(TensorInference; manual = false)

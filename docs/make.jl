using TensorInference
using TensorInference: OMEinsum
using TensorInference.OMEinsum: OMEinsumContractionOrders
using Documenter, Literate
using Pkg

# Literate Examples
const EXAMPLE_DIR = pkgdir(TensorInference, "examples")
const LITERATE_GENERATED_DIR = pkgdir(TensorInference, "docs", "src", "generated")
mkpath(LITERATE_GENERATED_DIR)
for each in readdir(EXAMPLE_DIR)
    # setup directory
    workdir = joinpath(LITERATE_GENERATED_DIR, each)
    cp(joinpath(EXAMPLE_DIR, each), workdir; force=true)
    # NOTE: for convenience, we use the `docs` environment
    # enter env
    # Pkg.activate(workdir)
    # Pkg.instantiate()
    # build
    input_file = joinpath(workdir, "main.jl")
    @info "building" input_file
    Literate.markdown(input_file, workdir; execute=true)
    # restore environment
    # Pkg.activate(Pkg.PREV_ENV_PATH[])
end

const EXTRA_JL = ["performance.jl"]
const SRC_DIR = pkgdir(TensorInference, "docs", "src")
for each in EXTRA_JL
    cp(joinpath(SRC_DIR, each), joinpath(LITERATE_GENERATED_DIR, each); force=true)
    input_file = joinpath(LITERATE_GENERATED_DIR, each)
    @info "building" input_file
    Literate.markdown(input_file, LITERATE_GENERATED_DIR; execute=true)
end

DocMeta.setdocmeta!(TensorInference, :DocTestSetup, :(using TensorInference); recursive=true)

makedocs(;
    modules=[TensorInference, OMEinsumContractionOrders],
    authors="Jin-Guo Liu, Martin Roa Villescas",
    repo="https://github.com/TensorBFS/TensorInference.jl/blob/{commit}{path}#{line}",
    sitename="TensorInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TensorBFS.github.io/TensorInference.jl",
        edit_link="main",
        assets = [joinpath("assets", "favicon.ico")],
    ),
    pages=[
        "Home" => "index.md",
        "Background" => [
            "Probabilistic Inference" => "probabilisticinference.md",
            "Tensor Networks" => "tensornetwork.md",
            "UAI file formats" => "uai-file-formats.md"
        ],
        "Examples" => [
            "Overview" => "examples-overview.md",
            "Asia Network" => "generated/asia/main.md",
            "Hard-core Lattice Gas" => "generated/hard-core-lattice-gas/main.md",
           ],
        "Performance tips" => "generated/performance.md",
        "API" => [
            "Public" => "api/public.md",
            "Internal" => "api/internal.md"
        ],
        "Contributing" => "contributing.md",
    ],
    doctest = false,
)

deploydocs(;
    repo="github.com/TensorBFS/TensorInference.jl",
    devbranch="main",
)

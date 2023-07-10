using TensorInference
using TensorInference: OMEinsum
using TensorInference.OMEinsum: OMEinsumContractionOrders
using Documenter, Literate

# Literate
const EXAMPLE_DIR = pkgdir(TensorInference, "examples")
const LITERATE_GENERATED_DIR = pkgdir(TensorInference, "docs", "src", "generated")
for each in readdir(EXAMPLE_DIR)
    workdir = joinpath(LITERATE_GENERATED_DIR, each)
    cp(joinpath(EXAMPLE_DIR, each), workdir; force=true)
    input_file = joinpath(workdir, "main.jl")
    @info "building" input_file
    Literate.markdown(input_file, workdir; execute=true)
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
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Asia network" => "generated/asia/main.md",
           ],
        "Performance Tips" => "performance.md",
        "References" => "ref.md",
    ],
    doctest = false,
)

deploydocs(;
    repo="github.com/TensorBFS/TensorInference.jl",
    devbranch="main",
)

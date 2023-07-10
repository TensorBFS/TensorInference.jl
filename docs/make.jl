using TensorInference
using TensorInference: OMEinsum, OMEinsumContractionOrders
using Documenter, Literate

# Literate
for each in readdir(pkgdir(TensorInference, "examples"))
    input_file = pkgdir(TensorInference, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(TensorInference, "docs", "src", "generated")
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
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
            "Asia network" => "generated/asia.md",
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

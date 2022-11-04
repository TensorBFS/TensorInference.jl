using TensorInference
using Documenter

DocMeta.setdocmeta!(TensorInference, :DocTestSetup, :(using TensorInference); recursive=true)

makedocs(;
    modules=[TensorInference],
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
    ],
)

deploydocs(;
    repo="github.com/TensorBFS/TensorInference.jl",
    devbranch="main",
)

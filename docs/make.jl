using ProbabilisticTensorNetworks
using Documenter

DocMeta.setdocmeta!(ProbabilisticTensorNetworks, :DocTestSetup, :(using ProbabilisticTensorNetworks); recursive=true)

makedocs(;
    modules=[ProbabilisticTensorNetworks],
    authors="Jin-Guo Liu, Martin Roa Villescas",
    repo="https://github.com/mroavi/ProbabilisticTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="ProbabilisticTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mroavi.github.io/ProbabilisticTensorNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mroavi/ProbabilisticTensorNetworks.jl",
    devbranch="main",
)

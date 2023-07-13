<p align="center">
<img width="700px" src="./docs/src/assets/logo-with-name.svg"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/dev/)
[![Build Status](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TensorBFS/TensorInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/TensorInference.jl)

This package implements a tensor network based probabilistic modeling toolbox, which covers the probability inference functionalities in [this page](https://uaicompetition.github.io/uci-2022/competition-entry/tasks/):
* PR: computing the partition function or probability of evidence,
* MAR: computing the marginal probability distribution over all variables given evidence.
* MAP: computing the most likely assignment to all variables given evidence.
* MMAP: computing the most likely assignment to the query variables after marginalizing out the remaining variables.

## Installation
<p>
<code>TensorInference</code> is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install <code>TensorInference</code>,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type
</p>

```julia
pkg> add TensorInference
```

To update, just type `up` in the package mode.

## Examples
Examples are in the [examples](examples) folder, which contains the following list of example problems
- [asia network](examples/asia)


## Supporting and Citing

Much of the software in this ecosystem was developed as a part of an academic research project.
If you would like to help support it, please star the repository.
If you use our software as part of your research, teaching, or other activities, we would like to request you to cite our [work (TBA)](). 
The
[CITATION.bib](CITATION.bib) file in the root of this repository lists the relevant papers.

## Questions and Contributions

Please Open an [issue](https://github.com/TensorBFS/TensorInference.jl/issues) if you encounter any problems, or have any feature request.

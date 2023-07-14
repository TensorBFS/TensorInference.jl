<p align="center">
<img width="700px" src="./docs/src/assets/logo-with-name.svg"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/dev/)
[![Build Status](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TensorBFS/TensorInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/TensorInference.jl)

This package presents a tensor network-based probabilistic modeling toolbox for
probabilistic inference. It features solutions for the [probability inference
tasks](https://uaicompetition.github.io/uci-2022/competition-entry/tasks/) of
the [UAI inference competitions](https://uaicompetition.github.io/uci-2022/),
which include:

- **PR**: Computing the partition function or probability of evidence.
- **MAR**: Computing the marginal probability distribution over all variables
  given evidence.
- **MAP**: Computing the most likely assignment to all variables given evidence.
- **MMAP**: Computing the most likely assignment to the query variables after
  marginalizing out the remaining variables.

## Installation

<p>
<code>TensorInference</code> is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install it, start Julia's <a
    href="https://docs.julialang.org/en/v1/manual/getting-started/">REPL</a>,
    press the <kbd>]</kbd> key to start the <a
    href="https://docs.julialang.org/en/v1/stdlib/Pkg/">package mode</a>, and
    then type
</p>

```julia
pkg> add TensorInference
```

To update, type `up` in the package mode.

## Examples

Check out the [examples](examples) directory to learn how to use the API of
`TensorInference`.

## Supporting and Citing

Much of the software in this ecosystem was developed as a part of an academic
research project. If you would like to help support it, please star the
repository. If you use our software as part of your research, teaching, or other
activities, please cite our [work (TBA)](). The [CITATION.bib](CITATION.bib)
file in the root of this repository lists the relevant papers.

## Questions and Contributions

Please open an [issue](https://github.com/TensorBFS/TensorInference.jl/issues)
if you encounter any problems, or have any feature requests.

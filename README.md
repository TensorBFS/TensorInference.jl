<p align="center">
<img width="700px" src="./docs/src/assets/logo-with-name.svg"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TensorBFS.github.io/TensorInference.jl/dev/)
[![Build Status](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TensorBFS/TensorInference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TensorBFS/TensorInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/TensorInference.jl)

<p>
TensorInference is an open source &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia
    </a>
&nbsp; package for probabilistic inference over discrete graphical models. It
leverages tensor-based technology for efficiently solving various inference
tasks.
</p>

## Features

TensorInference supports finding solutions to the most common [probability
inference
tasks](https://uaicompetition.github.io/uci-2022/competition-entry/tasks/) of
the [UAI inference competitions](https://uaicompetition.github.io/uci-2022/),
which include: 

- **PR**: The partition function or probability of evidence
- **MAR**: The marginal probability distribution over all variables
  given evidence
- **MAP**: The most likely assignment to all variables given evidence
- **MMAP**: The most likely assignment to the query variables after
  marginalizing out the remaining variables

## Installation

Install TensorInference through the Julia package manager:

```julia
pkg> add TensorInference
```

## Examples

Usage examples can be found in the [examples](examples) folder, and for a
comprehensive introduction to the package read the
[documentation](https://TensorBFS.github.io/TensorInference.jl/stable/) .

## Citing

If you use TensorInference as part of your research, teaching, or other
activities, please consider citing the following publication: [(TBA)]().

## Questions and Contributions

Please open an [issue](https://github.com/TensorBFS/TensorInference.jl/issues)
if you encounter any problems, or have any feature requests.

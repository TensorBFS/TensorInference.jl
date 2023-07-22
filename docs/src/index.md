```@meta
CurrentModule = TensorInference
```

# TensorInference.jl

*TensorInference* is a standalone solver written in Julia, that harnesses
tensor-based technology to implement state-of-the-art algorithms for
probabilistic inference in graphical models. 

## Package features

Solutions to the most common probabilistic inference tasks, including:

- **Probability of evidence (PR)**: Calculates the total probability of the
  observed evidence across all possible states of the unobserved variables.

- **Marginal inference (MAR)**: Computes the probability distribution of a
  subset of variables, ignoring the states of all other variables.

- **Maximum a Posteriori Probability estimation (MAP)**: Finds the most probable
  state of a subset of unobserved variables given some observed evidence.

- **Marginal Maximum a Posteriori (MMAP)**: Finds the most probable state of a
  subset of variables, averaging out the uncertainty over the remaining ones.

- **Sampling from distribution**: Generates samples from the probabilistic model.

## Outline
```@contents
Pages = [
  "background.md",
  "examples-overview.md",
  "uai-file-formats.md",
  "performance.md",
  "api/public.md",
  "api/internal.md",
]
Depth = 1
```

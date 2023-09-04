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

## Why TensorInference.jl

A major challenge in developing intelligent systems is the ability to reason
under uncertainty, a challenge that appears in many real-world problems across
various domains, including artificial intelligence, medical diagnosis,
computer vision, computational biology, and natural language processing.
Reasoning under uncertainty involves calculating the probabilities of relevant
variables while taking into account any information that is acquired. This
process, which can be thought of as drawing global insights from local
observations, is known as *probabilistic inference*.

*Probabilistic graphical models* (PGMs) provide a unified framework to perform
probabilistic inference. These models use graphs to represent the joint
probability distribution of complex systems in a concise manner by exploiting
the conditional independence between variables in the model. Additionally,
they form the foundation for various algorithms that enable efficient
probabilistic inference.

However, even with the representational aid of PGMs, performing probabilistic
inference remains an intractable endeavor on many real-world models. The
reason is that performing probabilistic inference involves complex
combinatorial optimization problems in very high dimensional spaces. To tackle
these challenges, more efficient and scalable inference algorithms are needed.

As an attempt to tackle the aforementioned challenges, we present
`TensorInference.jl`, a Julia package for probabilistic inference that
combines the representational capabilities of PGMs with the computational
power of tensor networks. By harnessing the best of both worlds,
`TensorInference.jl` aims to enhance the performance of probabilistic
inference, thereby expanding the tractability spectrum of exact inference for
more complex, real-world models.

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

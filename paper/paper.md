---
title: 'TensorInference: A Julia package for tensor-based probabilistic
inference'
#title: 'TensorInference: A Julia package for probabilistic inference through
#tensor-based technology'
#title: 'TensorInference: A Julia package for probabilistic inference using
#tensor networks'
tags:
  - Julia
  - probabilistic graphical models
  - tensor networks
  - probabilistic inference
authors:
  - name: Martin Roa-Villescas
    orcid: 0009-0009-0291-503X
    equal-contrib: true
    affiliation: 1
  - name: Jin-Guo Liu
    orcid: 0000-0003-1635-2679
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Eindhoven University of Technology
   index: 1
 - name: Hong Kong University of Science and Technology (Guangzhou)
   index: 2
date: 18 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.

# Citations to entries in paper.bib should be in
# [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
# format.
# 
# If you want to cite a software repository URL (e.g. something on GitHub without a preferred
# citation) then you can do it with the example BibTeX entry below for @fidgit.
# 
# For a quick reference, the following citation commands can be used:
# - `@author:2001`  ->  "Author et al. (2001)"
# - `[@author:2001]` -> "(Author et al., 2001)"
# - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures can be included like this:
# ![Caption for example figure.\label{fig:example}](figure.png)
# and referenced from text using \autoref{fig:example}.
# 
# Figure sizes can be customized by adding an optional second parameter:
# ![Caption for example figure.](figure.png){ width=20% }
---

# Summary

`TensorInference.jl` is a Julia [@bezanson2017julia] package designed for
performing probabilistic inference in discrete graphical models. Capitalizing on
the recent advances in the field of tensor networks [@orus2014practical;
@orus2019tensor; @robeva2019duality], `TensorInference.jl` offers
high-performance solutions for prevalent inference problems. Specifically, it
provides methods to: 

1. calculate the partition function (also known as the probability of evidence).
2. compute the marginal probability distribution over each variable given
   evidence.
3. find the most likely assignment to all variables given evidence.
4. find the most likely assignment to a set of query variables after
   marginalizing out the remaining variables.
5. draw samples from the posterior distribution given evidence
   [@han2018unsupervised; @cheng2019tree].

The use of a tensor network-based infrastructure [@fishman2022itensor;@Jutho2023] offers several advantages when
dealing with complex computational tasks. Firstly, it simplifies the process of
computing gradients by employing differentiable programming
[@liao2019differentiable], a critical operation for the aforementioned inference
tasks. Secondly, it supports generic element types without a significant
compromise on performance. The advantage of supporting generic element types
lies in the ability to solve a variety of problems using the same tensor network
contraction algorithm, simply by varying the element types used. This
flexibility has allowed us to seamlessly implement solutions for several of the
inference tasks mentioned earlier [@liu2021tropical;@liu2022computing].
Thirdly, it allows users to define a hyper-optimized contraction order, which is
known to have a significant impact on the computational performance of
contracting tensor networks [@markov2008simulating;@Pan2022;@gao2021limitations].
`TensorInference.jl` provides a predefined set of
state-of-the-art contraction ordering methods. These methods include a *local
search based method* (`TreeSA`) [@kalachev2022multitensor], two *min-cut based
methods* (`KaHyParBipartite`) [@gray2021hyper] and (`SABipartite`), and a
*greedy method* (`GreedyMethod`). Finally, tensor networks -- and by extension,
`TensorInference.jl` -- harness the latest developments in computational
technology, including a highly optimized set of BLAS routines
[@blackford2002updated] and GPU technology.

# Statement of need

A major challenge in developing intelligent systems is the ability to reason
under uncertainty, a challenge that appears in many real-world problems across
various domains, including artificial intelligence, medical diagnosis, computer
vision, computational biology, and natural language processing. Reasoning under
uncertainty involves calculating the probabilities of relevant variables while
taking into account any information that is acquired. This process, which can be
thought of as drawing global insights from local observations, is known as
*probabilistic inference*.

*Probabilistic graphical models* (PGMs) provide a unified framework to perform
probabilistic inference. These models use graphs to represent the joint
probability distribution of complex systems in a concise manner by exploiting
the conditional independence between variables in the model. Additionally, they
form the foundation for various algorithms that enable efficient probabilistic
inference.

However, even with the representational aid of PGMs, performing probabilistic
inference remains an intractable endeavor on many real-world models. The reason
is that performing probabilistic inference involves complex combinatorial
optimization problems in very high dimensional spaces. To tackle these
challenges, more efficient and scalable inference algorithms are needed.

As an attempt to tackle the aforementioned challenges, we present
`TensorInference.jl`, a Julia package for probabilistic inference that combines
the representational capabilities of PGMs with the computational power of tensor
networks. By harnessing the best of both worlds, `TensorInference.jl` aims to
enhance the performance of probabilistic inference, thereby expanding the
tractability spectrum of exact inference for more complex, real-world models.

`TensorInference.jl` succeeds `JunctionTrees.jl` [@roa2022partial;
@roa2023scaling], a Julia package implementing the Junction Tree Algorithm (JTA)
[@lauritzen1988local; @jensen1990bayesian]. While the latter employs
tensor-based technology to optimize the computation of individual sum-product
messages within the JTA context, `TensorInference.jl` takes a different route.
It adopts a holistic tensor network approach, which opens new doors for
optimization opportunities, and significantly reduces the algorithm's complexity
compared to the JTA.

# Usage example

The graph below corresponds to the *ASIA network* [@lauritzen1988local], a
simple Bayesian network [@pearl1985bayesian] used extensively in educational
settings. It describes the probabilistic relationships between different random
variables which correspond to possible diseases, symptoms, risk factors and test
results.

![The ASIA network: a simplified example of a Bayesian network from the context
of medical diagnosis [@lauritzen1988local].
](./figures/asia-network/out/asia-network.pdf)

In the example, a patient has recently visited Asia and is now experiencing
dyspnea. These conditions serve as the evidence for the observed variables ($A$
and $D$). The doctor's task is to assess the likelihood of various diseases —-
tuberculosis, lung cancer, and bronchitis -- which constitute the query
variables in this scenario ($T$, $L$, and $B$).

We now demonstrate how to use `TensorInference.jl` for conducting a variety of
inference tasks on this toy example. Please note that as the API may evolve, we
recommend checking the
[examples](https://github.com/TensorBFS/TensorInference.jl/tree/main/examples)
directory of the official `TensorInference.jl` repository for the most
up-to-date version of this example.

```{=latex}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},fontsize=\small}
\newcommand{\InTok}[1]{\textcolor[rgb]{0.0, 0.0, 0.5}{#1}} % mrv
\newcommand{\OutTok}[1]{\textcolor[rgb]{0.545, 0.0, 0.0}{#1}} % mrv
```

```julia
# Import the TensorInference package, which provides the functionality needed
# for working with tensor networks and probabilistic graphical models.
using TensorInference

# Load the ASIA network model from the `asia.uai` file located in the examples
# directory. Refer to the documentation of this package for a description of the
# format of this file.
instance = read_instance(pkgdir(TensorInference, "examples", "asia", "asia.uai"))

# Create a tensor network representation of the loaded model.
tn = TensorNetworkModel(instance)

TensorNetworkModel{Int64, OMEinsum.DynamicNestedEinsum{Int64}, Array{Float64}}
variables: 1, 2, 3, 4, 5, 6, 7, 8
contraction time = 2^6.044, space = 2^2.0, read-write = 2^7.098

# Calculate the log_10 partition function.
probability(tn) |> first |> log10

0.0

# Calculate the marginal probabilities of each random variable in the model.
marginals(tn)

8-element Vector{Vector{Float64}}:
 [0.01, 0.99]
 [0.0104, 0.9895999999999999]
 [0.5, 0.49999999999999994]
 [0.055000000000000014, 0.9450000000000001]
 [0.44999999999999996, 0.5499999999999999]
 [0.06482800000000002, 0.9351720000000001]
 [0.11029004000000002, 0.88970996]
 [0.43597060000000004, 0.5640294]

# Set the evidence: We assume that the "X-ray" result (variable 7) is positive.
set_evidence!(instance, 7 => 0)

# Since setting the evidence may affect the contraction order of the tensor
# network, we need to recompute it.
tn = TensorNetworkModel(instance)

# Calculate the maximum log-probability among all configurations.
maximum_logp(tn)

0-dimensional Array{Float64, 0}:
-3.6522217920023303

# Generate 10 samples from the probability distribution represented by the model.
sample(tn, 10)

8×10 Matrix{Int64}:
 1  1  1  1  1  1  1  1  1  1
 1  1  1  1  1  1  1  1  1  0
 1  0  0  0  1  0  0  0  0  0
 1  1  0  0  1  1  0  0  0  1
 1  0  1  0  0  1  1  0  0  0
 1  1  0  0  1  1  0  0  0  0
 0  0  0  0  0  0  0  0  0  0
 1  0  1  1  0  0  1  0  0  0

# Retrieve both the maximum log-probability and the most probable configuration.
# In this configuration, the most likely outcomes are that the
# patient smokes (variable 3) and has lung cancer (variable 4).
logp, cfg = most_probable_config(tn)

(-3.6522217920023303, [1, 1, 0, 0, 0, 0, 0, 0])

# Compute the most probable values of certain variables (e.g., 4 and 7) while
# marginalizing over others. This is known as Maximum a Posteriori (MAP)
# estimation.
set_query!(instance, [4, 7])
mmap = MMAPModel(instance)

MMAPModel{Int64, Array{Float64}}
variables: 4, 7 (evidence → 0)
query variables: [[1, 2, 6, 5, 3, 8]]
contraction time = 2^6.022, space = 2^2.0, read-write = 2^7.033

# Get the most probable configurations for variables 4 and 7.
most_probable_config(mmap)

(-2.8754627318176693, [1, 0])

# Compute the total log-probability of having lung cancer. The results suggest
# that the probability is roughly half.
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])

(-2.8754627318176693, -2.9206248010671856)
```

# Acknowledgments

This work is partially funded by the Netherlands Organization for Scientific
Research and the Guangzhou Municipal Science and Technology Project (No.
2023A03J0003). We extend our gratitude to Madelyn Cain and Patrick Wijnings for
their insightful discussions on the intersection of tensor networks and
probabilistic graphical models.

# References

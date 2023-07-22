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
performing probabilistic inference in discrete graphical models. It leverages
the recent explosion of advances in the field of tensor networks
[@orus2014practical; @orus2019tensor; @robeva2019duality] to provide
high-performance solutions for common inference problems. Specifically,
`TensorInference.jl` offers mechanisms to: 

1. calculate the partition function (also known as the probability of evidence).
2. compute the marginal probability distribution over each variable given
   evidence.
3. find the most likely assignment to all variables given evidence.
4. find the most likely assignment to a set of query variables after
   marginalizing out the remaining variables.
5. draw samples from the posterior distribution given evidence
   [@han2018unsupervised; @cheng2019tree].

The infrastructure based on tensor networks introduces several benefits in
handling complex computational tasks. First, it provides a convenient approach
to differentiate a tensor network program [@liao2019differentiable], a crucial
operation in the computation of the inference tasks listed above. Second, it
supports generic element types without sacrificing significant performance. The
advantage of this generic element type support is that solutions to diverse
problems can be obtained using the same tensor network contraction algorithm but
with different element types. This introduces a level of flexibility and
adaptability that can handle a broad spectrum of problem domains efficiently
[@liu2021tropical; @liu2022computing]. Third, it allows users to define a
hyper-optimized contraction order, which is known to have a significant impact
on the computational performance of contracting tensor networks
[@markov2008simulating; @pan2021simulating; @gao2021limitations].
`TensorInference.jl` makes a predefined set of state-of-the-art contraction
ordering methods available to the users. These methods include a *local search
based method* (`TreeSA`) [@kalachev2022multitensor], two *min-cut based methods*
(`KaHyParBipartite`) [@gray2021hyper] and (`SABipartite`), and a *greedy method*
(`GreedyMethod`). Finally, `TensorInference.jl` harnesses the latest
developments in computational technology, including a highly optimized set of
BLAS [@blackford2002updated] routines and GPU technology.

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
probability distribution of complex systems concisely by exploiting the
conditional independence between variables in the model. Additionally, they form
the foundation for various algorithms that enable efficient probabilistic
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
[@lauritzen1988local; @jensen1990bayesian]. While the latter optimizes
computation of individual sum-product messages within the JTA context by
employing tensor-based technology at the backend level, `TensorInference.jl`
takes a different route. It adopts a holistic tensor network approach, fully
integrating the JTA, significantly reducing the algorithm's complexity, and
thereby opening new doors for optimization opportunities.

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
and $D$). The doctor's task is to assess the likelihood of various diseases —
tuberculosis, lung cancer, and bronchitis - which constitute the query variables
in this scenario ($T$, $L$, and $B$).

We now demonstrate how to use `TensorInference.jl` for conducting a variety of
inference tasks on this toy example. Please note that as the API may evolve, we
recommend checking the
[examples](https://github.com/TensorBFS/TensorInference.jl/tree/main/examples/asia)
directory of the official `TensorInference.jl` repository for the most
up-to-date version of this example.

```julia

# Import the TensorInference package, which provides the functionality needed
# for working with tensor networks and probabilistic graphical models.
using TensorInference

# Load the ASIA network model from the `asia.uai` file located in the examples
# directory. Refer to the documentation of this package for a description of the
# format of this file.
instance = read_instance(pkgdir(TensorInference, "examples", "asia", "asia.uai"))

# Create a tensor network representation of the loaded model.
# The variable 7 is the variable of interest, which will be retained in the output.
tn = TensorNetworkModel(instance; openvars=[7])

# Calculate the partition function for each assignment of variable 7.
probability(tn)

# Calculate the marginal probabilities of each random variable in the model.
marginals(tn)

# Retrieve the variables associated with the tensor network model.
get_vars(tn)

# Assume that the "X-ray" result (variable 7) is positive.
# Since setting an evidence may affect the contraction order of the tensor
# network, recompute it.
tn = TensorNetworkModel(instance; evidence=Dict(7 => 0))

# Calculate the maximum log-probability among all configurations.
maximum_logp(tn)

# Generate 10 samples from the probability distribution represented by the
# model.
sample(tn, 10)

# Retrieve both the maximum log-probability and the most probable
# configuration. In this configuration, the most likely outcomes are that the
# patient smokes (variable 3) and has lung cancer (variable 4).
logp, cfg = most_probable_config(tn)

# Compute the most probable values of certain variables (e.g., 4 and 7) while
# marginalizing over others. This is known as Maximum a Posteriori (MAP)
# estimation.
mmap = MMAPModel(instance, queryvars=[4, 7])

# Get the most probable configurations for variables 4 and 7.
most_probable_config(mmap)

# Compute the total log-probability of having lung cancer. The results suggest
# that the probability is roughly half.
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])
```

# Acknowledgments

This work is partially funded by the Netherlands Organization for Scientific
Research and the Guangzhou Municipal Science and Technology Project (No.
2023A03J0003). We extend our gratitude to Madelyn Cain and Patrick Wijnings for
their insightful discussions on the intersection of tensor networks and
probabilistic graphical models.

# References

---
title: 'TensorInference: A Julia package for probabilistic inference through
tensor-based technology'
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

`TensorInference.jl` is a Julia [@bezanson2017julia] library designed for
performing probabilistic inference in discrete graphical models. It leverages
the recent explosion of advances in the field of tensor networks to provide
high-performance solutions for common inference tasks. These tasks include
calculating: 1) the partition function or probability of evidence, 2) the
marginal probability distribution over each variable given evidence, 3) the most
likely assignment to all variables given evidence, and 4) the most likely
assignment to the query variables after marginalizing out the remaining
variables. The infrastructure based on tensor networks allows users to define
the contraction ordering method, which is known to have a significant impact on
the computational performance of these algorithms. A predefined set of
state-of-the-art contraction ordering methods is made available to users. These
methods include the *recursive multi-tensor contraction method* (`TreeSA`)
[@kalachev2022multitensor], the *hyper-optimized tensor network contraction
method* (`KaHyParBipartite`) [@gray2021hyper], the *hierarchical partitioning
with dynamic slicing method* (`SABipartite`) [@pan2021simulating], and a
*greedy-based memory minimization method* (`GreedyMethod`) [@liu2022computing].
Finally, `TensorInference.jl` leverages the latest developments in computational
technology, including a highly optimized set of BLAS routines and GPU
technology.

# Statement of need

A major challenge in developing intelligent systems is the ability to reason
under uncertainty, a challenge that appears in many real-world problems across
various domains, including artificial intelligence, medical diagnosis, computer
vision, computational biology, and natural language processing. Reasoning under
uncertainty involves drawing global insights from local observations, a process
known as *probabilistic inference*.

*Probabilistic graphical models* (PGMs) provide a unified framework to address
these challenges. These models use graphs to concisely represent the joint
probability distribution of complex systems by exploiting the conditional
independence between variables in the model. Additionally, they form the
foundation for various algorithms that enable efficient probabilistic inference.

However, performing probabilistic inference on many real-world problems remains
intractable due to the high dimensional spaces and lack of structure in their
data. To address these problems, more efficient and scalable inference
algorithms are needed.

We present `TensorInference.jl`, a Julia [@bezanson2012julia;
@bezanson2017julia] package for probabilistic inference that combines the
representational capabilities of PGMs with the computational power of tensor
networks.

# Usage example

The graph below corresponds to the *ASIA network*, a simple Bayesian model
used extensively in educational settings. It was introduced by Lauritzen in
[@lauritzen1988local].

![](./figures/asia-network/out/asia-network.pdf)

We now demonstrate how to use the TensorInference.jl package for conducting a
variety of inference tasks on this toy example.

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

# Calculate the log10 partition function 
probability(tn) |> first |> log10

# Calculate the marginal probabilities of each random variable in the model.
marginals(tn)

# Retrieve the variables associated with the tensor network model.
get_vars(tn)

# Set an evidence: Assume that the "X-ray" result (variable 7) is positive.
set_evidence!(instance, 7 => 0)

# Since setting an evidence may affect the contraction order of the tensor
# network, recompute it.
tn = TensorNetworkModel(instance)

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
set_query!(instance, [4, 7])
mmap = MMAPModel(instance)

# Get the most probable configurations for variables 4 and 7.
most_probable_config(mmap)

# Compute the total log-probability of having lung cancer. The results suggest
# that the probability is roughly half.
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])
```

# Acknowledgments

This work is partially funded by the Netherlands Organization for Scientific
Research. The authors want to thank Madelyn Cain for helpful advice.

# References

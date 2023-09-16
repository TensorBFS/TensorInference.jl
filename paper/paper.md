---
title: 'TensorInference: A Julia package for tensor-based probabilistic
inference'
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

Probabilistic inference entails the process of drawing conclusions from
observed data through the axioms of probability theory. Inference algorithms
fall into two broad categories: *exact* and *approximate* methods. The main
challenge in applying exact inference to real-world problems is its NP-hard
computational complexity tied to the model's *treewidth*, a metric of network
connectivity. This has prompted a research shift to approximate methods like
*Markov chain Monte Carlo* and *variational* inference. Prominent examples of
packages that implement such algorithms include `Stan` [@carpenter2017stan],
`PyMC3` [@oriol2023pymc], `Turing.jl` [@ge2018turing], and `RxInfer.jl`
[@bagaev2023rxinfer]. However, while these methods offer superior scalability,
they do not provide formal guarantees of accuracy --- a challenge that is, in
itself, NP-hard to address. Consequently, exact inference methods are gaining
renewed interest for their promise of higher accuracy.

`TensorInference.jl` is a Julia [@bezanson2017julia] package designed for
performing exact probabilistic inference in discrete graphical models. 
Capitalizing on
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

A *tensor* is a mathematical object that generalizes scalars, vectors, and
matrices to higher dimensions. In essence, it is a multi-dimensional array of
numbers, often used for representing complex data structures in physics,
engineering, computer science, and data analytics. A *tensor network* consists
of a set of tensors in which some or all indices are contracted according to a
specific pattern [@Jutho2023]. The term *contraction* refers to the summation
over all the possible values along one or more dimensions of a set of tensors.
These networks excel at capturing the correlations of different states in
complex systems.

The order in which tensor indices are contracted plays a pivotal role in
computational efficiency. Different contraction sequences can produce the same
mathematical outcome, but the computational costs can vary by orders of
magnitude. Since tensor network methods frequently involve multiple
contractions, optimizing the contraction order becomes crucial.

The use of a tensor network-based infrastructure
[@Jutho2023] offers several advantages when dealing with
complex computational tasks. Firstly, it simplifies the process of computing
gradients by employing differentiable programming [@liao2019differentiable], a
critical operation for the aforementioned inference tasks. Secondly, it
supports generic element types without a significant compromise on
performance. The advantage of supporting generic element types lies in the
ability to solve a variety of problems using the same tensor network
contraction algorithm, simply by varying the element types used. This
flexibility has allowed us to seamlessly implement solutions for several of
the inference tasks described above [@liu2021tropical;@liu2022computing].
Thirdly, it allows users to define a hyper-optimized contraction order, which
is known to have a significant impact on the computational performance of
contracting tensor networks
[@markov2008simulating;@Pan2022;@gao2021limitations]. `TensorInference.jl`
provides a predefined set of state-of-the-art contraction ordering methods.
These methods include a *local search based method* (`TreeSA`)
[@kalachev2022multitensor], two *min-cut based methods* (`KaHyParBipartite`)
[@gray2021hyper] and (`SABipartite`), and a *greedy method* (`GreedyMethod`).
Finally, `TensorInference.jl` leverages the cutting-edge developments commonly
found in tensor network libraries, including a highly optimized set of BLAS
routines [@blackford2002updated] and GPU technology.

`TensorInference.jl` succeeds `JunctionTrees.jl` [@roa2022partial;
@roa2023scaling], a Julia package implementing the Junction Tree Algorithm
(JTA) [@lauritzen1988local; @jensen1990bayesian]. While the latter employs
tensor-based technology to optimize the computation of individual sum-product
messages within the JTA context, `TensorInference.jl` takes a different route.
It adopts a holistic tensor network approach, which opens new doors for
optimization opportunities and significantly reduces the algorithm's
complexity compared to the JTA. Other prominent examples of exact inference
packages for probabilistic inference include `libDAI` [@mooij2010libdai],
`Merlin` [@marinescu2022merlin], and `toulbar2` [@hurley2016multi]. For a
performance comparison of `TensorInference.jl` against these alternatives,
please see the [Performance
evaluation](https://tensorbfs.github.io/TensorInference.jl/dev/performance-evaluation/)
section in the documentation of `TensorInference.jl`.

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
dyspnea. These conditions serve as the evidence for the observed variables
($A$ and $D$). The doctor's task is to assess the likelihood of various
diseases --- tuberculosis, lung cancer, and bronchitis --- which constitute
the query variables in this scenario ($T$, $L$, and $B$).

We now demonstrate how to use `TensorInference.jl` for conducting a variety of
inference tasks on this toy example. Please note that as the API may evolve, we
recommend checking the
[examples](https://github.com/TensorBFS/TensorInference.jl/tree/main/examples)
directory of the official `TensorInference.jl` repository for the most
up-to-date version of this example.

```julia
```

```{=latex}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},fontsize=\small}
\newcommand{\InTok}[1]{\textcolor[rgb]{0.0, 0.0, 0.5}{#1}} % mrv
\newcommand{\OutTok}[1]{\textcolor[rgb]{0.545, 0.0, 0.0}{#1}} % mrv

\begin{Shaded}
\begin{Highlighting}[]
\CommentTok{\# Import the TensorInference package, which provides the functionality needed}
\CommentTok{\# for working with tensor networks and probabilistic graphical models.}
\InTok{In [1]: }\ImportTok{using} \BuiltInTok{TensorInference}

\CommentTok{\# Load the ASIA network model from \textasciigrave{}asia.uai\textasciigrave{} in the examples directory.}
\CommentTok{\# Refer to the package documentation for a description of the format of this file.}
\NormalTok{model }\OperatorTok{=} \FunctionTok{read\_model\_file}\NormalTok{(}\FunctionTok{pkgdir}\NormalTok{(TensorInference, }\StringTok{"examples"}\NormalTok{, }\StringTok{"asia"}\NormalTok{, }\StringTok{"asia.uai"}\NormalTok{))}

\CommentTok{\# Create a tensor network representation of the loaded model.}
\NormalTok{tn }\OperatorTok{=} \FunctionTok{TensorNetworkModel}\NormalTok{(model)}

\OutTok{Out [1]: }\NormalTok{TensorNetworkModel\{}\DataTypeTok{Int64}\NormalTok{, OMEinsum.DynamicNestedEinsum\{}\DataTypeTok{Int64}\NormalTok{\}, }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}
         \NormalTok{variables}\OperatorTok{:} \FloatTok{1}\NormalTok{, }\FloatTok{2}\NormalTok{, }\FloatTok{3}\NormalTok{, }\FloatTok{4}\NormalTok{, }\FloatTok{5}\NormalTok{, }\FloatTok{6}\NormalTok{, }\FloatTok{7}\NormalTok{, }\FloatTok{8}
         \NormalTok{contraction time }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{6.044}\NormalTok{, space }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{2.0}\NormalTok{, read}\OperatorTok{{-}}\NormalTok{write }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{7.098}

\CommentTok{\# Calculate the partition function. Since the factors in this model are}
\CommentTok{\# normalized, the partition function is the same as the total probability, 1.}
\InTok{In [2]: }\FunctionTok{probability}\NormalTok{(tn) }\OperatorTok{|\textgreater{}}\NormalTok{ first }

\OutTok{Out [2]: }\FloatTok{1.0000000000000002}

\CommentTok{\# Calculate the marginal probabilities of each random variable in the model.}
\InTok{In [3]: }\FunctionTok{marginals}\NormalTok{(tn)}

\OutTok{Out [3]: }\DataTypeTok{Dict}\NormalTok{\{}\DataTypeTok{Vector}\NormalTok{\{}\DataTypeTok{Int64}\NormalTok{\}, }\DataTypeTok{Vector}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\} with }\FloatTok{8}\NormalTok{ entries}\OperatorTok{:}
       \NormalTok{  [}\FloatTok{8}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.435971}\NormalTok{, }\FloatTok{0.564029}\NormalTok{]}
       \NormalTok{  [}\FloatTok{3}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.5}\NormalTok{, }\FloatTok{0.5}\NormalTok{]}
       \NormalTok{  [}\FloatTok{1}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.01}\NormalTok{, }\FloatTok{0.99}\NormalTok{]}
       \NormalTok{  [}\FloatTok{5}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.45}\NormalTok{, }\FloatTok{0.55}\NormalTok{]}
       \NormalTok{  [}\FloatTok{4}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.055}\NormalTok{, }\FloatTok{0.945}\NormalTok{]}
       \NormalTok{  [}\FloatTok{6}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.064828}\NormalTok{, }\FloatTok{0.935172}\NormalTok{]}
       \NormalTok{  [}\FloatTok{7}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.11029}\NormalTok{, }\FloatTok{0.88971}\NormalTok{]}
       \NormalTok{  [}\FloatTok{2}\NormalTok{] }\OperatorTok{=\textgreater{}}\NormalTok{ [}\FloatTok{0.0104}\NormalTok{, }\FloatTok{0.9896}\NormalTok{]}

\CommentTok{\# Set the evidence to assume that the \textquotesingle{}X{-}ray\textquotesingle{} result (variable 7) is positive.}
\CommentTok{\# Recompute the contraction order of the tensor network, as setting the evidence}
\CommentTok{\# may affect it.}
\InTok{In [4]: }\NormalTok{tn }\OperatorTok{=} \FunctionTok{TensorNetworkModel}\NormalTok{(model, evidence }\OperatorTok{=} \FunctionTok{Dict}\NormalTok{(}\FloatTok{7} \OperatorTok{=\textgreater{}} \FloatTok{0}\NormalTok{))}

\OutTok{Out [4]: }\NormalTok{TensorNetworkModel\{}\DataTypeTok{Int64}\NormalTok{, OMEinsum.DynamicNestedEinsum\{}\DataTypeTok{Int64}\NormalTok{\}, }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}
         \NormalTok{variables}\OperatorTok{:} \FloatTok{1}\NormalTok{, }\FloatTok{2}\NormalTok{, }\FloatTok{3}\NormalTok{, }\FloatTok{4}\NormalTok{, }\FloatTok{5}\NormalTok{, }\FloatTok{6}\NormalTok{, }\FloatTok{7}\NormalTok{ (evidence }\OperatorTok{→} \FloatTok{0}\NormalTok{), }\FloatTok{8}
         \NormalTok{contraction time }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{6.0}\NormalTok{, space }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{2.0}\NormalTok{, read}\OperatorTok{{-}}\NormalTok{write }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{7.066}

\CommentTok{\# Calculate the maximum log{-}probability among all configurations.}
\InTok{In [5]: }\FunctionTok{maximum\_logp}\NormalTok{(tn)}

\OutTok{Out [5]: }\FloatTok{0}\OperatorTok{{-}}\NormalTok{dimensional }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{, }\FloatTok{0}\NormalTok{\}}\OperatorTok{:}
         \OperatorTok{{-}}\FloatTok{3.6522217920023303}

\CommentTok{\# Generate 10 samples from the posterior distribution.}
\InTok{In [6]: }\FunctionTok{sample}\NormalTok{(tn, }\FloatTok{10}\NormalTok{)}

\OutTok{Out [6]: }\FloatTok{10}\OperatorTok{{-}}\NormalTok{element TensorInference.Samples\{}\DataTypeTok{Int64}\NormalTok{\}}\OperatorTok{:}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{]}
        \NormalTok{ [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{]}

\CommentTok{\# Retrieve both the maximum log{-}probability and the most probable}
\CommentTok{\# configuration. In this configuration, the most likely outcomes are that the}
\CommentTok{\# patient smokes (variable 3) and has lung cancer (variable 4).}
\InTok{In [7]: }\NormalTok{logp, cfg }\OperatorTok{=} \FunctionTok{most\_probable\_config}\NormalTok{(tn)}

\OutTok{Out [7]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{3.6522217920023303}\NormalTok{, [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\CommentTok{\# Compute the most probable values for a subset of variables (e.g., 4 and 7)}
\CommentTok{\# while marginalizing over the others. This process is known as Maximum a}
\CommentTok{\# Posteriori (MAP) estimation.}
\InTok{In [8]: }\NormalTok{mmap }\OperatorTok{=} \FunctionTok{MMAPModel}\NormalTok{(model, evidence}\OperatorTok{=}\FunctionTok{Dict}\NormalTok{(}\FloatTok{7}\OperatorTok{=\textgreater{}}\FloatTok{0}\NormalTok{), queryvars}\OperatorTok{=}\NormalTok{[}\FloatTok{4}\NormalTok{,}\FloatTok{7}\NormalTok{])}

\OutTok{Out [8]: }\NormalTok{MMAPModel\{}\DataTypeTok{Int64}\NormalTok{, }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}
         \NormalTok{variables}\OperatorTok{:} \FloatTok{4}\NormalTok{, }\FloatTok{7}\NormalTok{ (evidence }\OperatorTok{→} \FloatTok{0}\NormalTok{)}
         \NormalTok{query variables}\OperatorTok{:}\NormalTok{ [[}\FloatTok{1}\NormalTok{, }\FloatTok{2}\NormalTok{, }\FloatTok{6}\NormalTok{, }\FloatTok{5}\NormalTok{, }\FloatTok{3}\NormalTok{, }\FloatTok{8}\NormalTok{]]}
         \NormalTok{contraction time }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{6.0}\NormalTok{, space }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{2.0}\NormalTok{, read}\OperatorTok{{-}}\NormalTok{write }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{7.0}

\CommentTok{\# Get the most probable configurations for variables 4 and 7.}
\InTok{In [9]: }\FunctionTok{most\_probable\_config}\NormalTok{(mmap)}

\OutTok{Out [9]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{2.8754627318176693}\NormalTok{, [}\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\CommentTok{\# Compute the total log{-}probability of having lung cancer. The results suggest}
\CommentTok{\# that the probability is roughly half.}
\InTok{In [10]: }\FunctionTok{log\_probability}\NormalTok{(mmap, [}\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{]), }\FunctionTok{log\_probability}\NormalTok{(mmap, [}\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\OutTok{Out [10]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{2.8754627318176693}\NormalTok{, }\OperatorTok{{-}}\FloatTok{2.920624801067186}\NormalTok{)}
\end{Highlighting}
\end{Shaded}
```

# Acknowledgments

This work is partially funded by the Netherlands Organization for Scientific
Research and the Guangzhou Municipal Science and Technology Project (No.
2023A03J0003). We extend our gratitude to Madelyn Cain and Patrick Wijnings for
their insightful discussions on the intersection of tensor networks and
probabilistic graphical models.

# References

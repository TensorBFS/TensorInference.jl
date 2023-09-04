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

The use of a tensor network-based infrastructure
[@fishman2022itensor;@Jutho2023] offers several advantages when dealing with
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
@roa2023scaling], a Julia package implementing the Junction Tree Algorithm (JTA)
[@lauritzen1988local; @jensen1990bayesian]. While the latter employs
tensor-based technology to optimize the computation of individual sum-product
messages within the JTA context, `TensorInference.jl` takes a different route.
It adopts a holistic tensor network approach, which opens new doors for
optimization opportunities, and significantly reduces the algorithm's complexity
compared to the JTA.

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

\CommentTok{\# Load the ASIA network model from the \textasciigrave{}asia.uai\textasciigrave{} file located in the examples}
\CommentTok{\# directory. Refer to the documentation of this package for a description of the}
\CommentTok{\# format of this file.}
\NormalTok{instance }\OperatorTok{=} \FunctionTok{read\_instance}\NormalTok{(}\FunctionTok{pkgdir}\NormalTok{(TensorInference, }\StringTok{"examples"}\NormalTok{, }\StringTok{"asia"}\NormalTok{, }\StringTok{"asia.uai"}\NormalTok{))}

\CommentTok{\# Create a tensor network representation of the loaded model.}
\NormalTok{tn }\OperatorTok{=} \FunctionTok{TensorNetworkModel}\NormalTok{(instance)}

\OutTok{Out [1]: }\NormalTok{TensorNetworkModel\{}\DataTypeTok{Int64}\NormalTok{, OMEinsum.DynamicNestedEinsum\{}\DataTypeTok{Int64}\NormalTok{\}, }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}
         \NormalTok{variables}\OperatorTok{:} \FloatTok{1}\NormalTok{, }\FloatTok{2}\NormalTok{, }\FloatTok{3}\NormalTok{, }\FloatTok{4}\NormalTok{, }\FloatTok{5}\NormalTok{, }\FloatTok{6}\NormalTok{, }\FloatTok{7}\NormalTok{, }\FloatTok{8}
         \NormalTok{contraction time }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{6.044}\NormalTok{, space }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{2.0}\NormalTok{, read}\OperatorTok{{-}}\NormalTok{write }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{7.098}

\CommentTok{\# Calculate the log\_10 partition function.}
\InTok{In [2]: }\FunctionTok{probability}\NormalTok{(tn) }\OperatorTok{|\textgreater{}}\NormalTok{ first }\OperatorTok{|\textgreater{}}\NormalTok{ log10}

\OutTok{Out [2]: }\FloatTok{0.0}

\CommentTok{\# Calculate the marginal probabilities of each random variable in the model.}
\InTok{In [3]: }\FunctionTok{marginals}\NormalTok{(tn)}

\OutTok{Out [3]: }\FloatTok{8}\OperatorTok{{-}}\NormalTok{element }\DataTypeTok{Vector}\NormalTok{\{}\DataTypeTok{Vector}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}\OperatorTok{:}
        \NormalTok{ [}\FloatTok{0.01}\NormalTok{, }\FloatTok{0.99}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.0104}\NormalTok{, }\FloatTok{0.9895999999999999}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.5}\NormalTok{, }\FloatTok{0.49999999999999994}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.055000000000000014}\NormalTok{, }\FloatTok{0.9450000000000001}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.44999999999999996}\NormalTok{, }\FloatTok{0.5499999999999999}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.06482800000000002}\NormalTok{, }\FloatTok{0.9351720000000001}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.11029004000000002}\NormalTok{, }\FloatTok{0.88970996}\NormalTok{]}
        \NormalTok{ [}\FloatTok{0.43597060000000004}\NormalTok{, }\FloatTok{0.5640294}\NormalTok{]}

\CommentTok{\# Set the evidence: We assume that the "X{-}ray" result (variable 7) is positive.}
\InTok{In [4]: }\FunctionTok{set\_evidence!}\NormalTok{(instance, }\FloatTok{7} \OperatorTok{=\textgreater{}} \FloatTok{0}\NormalTok{)}

\CommentTok{\# Since setting the evidence may affect the contraction order of the tensor}
\CommentTok{\# network, we need to recompute it.}
\NormalTok{tn }\OperatorTok{=} \FunctionTok{TensorNetworkModel}\NormalTok{(instance)}

\CommentTok{\# Calculate the maximum log{-}probability among all configurations.}
\FunctionTok{maximum\_logp}\NormalTok{(tn)}

\OutTok{Out [4]: }\FloatTok{0}\OperatorTok{{-}}\NormalTok{dimensional }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{, }\FloatTok{0}\NormalTok{\}}\OperatorTok{:}
         \OperatorTok{{-}}\FloatTok{3.6522217920023303}

\CommentTok{\# Generate 10 samples from the probability distribution represented by the model.}
\InTok{In [5]: }\FunctionTok{sample}\NormalTok{(tn, }\FloatTok{10}\NormalTok{)}

\OutTok{Out [5]: }\FloatTok{8}\OperatorTok{×}\FloatTok{10} \DataTypeTok{Matrix}\NormalTok{\{}\DataTypeTok{Int64}\NormalTok{\}}\OperatorTok{:}
        \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}
        \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{1}  \FloatTok{0}
        \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}
        \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}
        \FloatTok{1}  \FloatTok{0}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}
        \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}
        \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}
        \FloatTok{1}  \FloatTok{0}  \FloatTok{1}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{1}  \FloatTok{0}  \FloatTok{0}  \FloatTok{0}

\CommentTok{\# Retrieve both the maximum log{-}probability and the most probable configuration.}
\CommentTok{\# In this configuration, the most likely outcomes are that the}
\CommentTok{\# patient smokes (variable 3) and has lung cancer (variable 4).}
\InTok{In [6]: }\NormalTok{logp, cfg }\OperatorTok{=} \FunctionTok{most\_probable\_config}\NormalTok{(tn)}

\OutTok{Out [6]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{3.6522217920023303}\NormalTok{, [}\FloatTok{1}\NormalTok{, }\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\CommentTok{\# Compute the most probable values of certain variables (e.g., 4 and 7) while}
\CommentTok{\# marginalizing over others. This is known as Maximum a Posteriori (MAP)}
\CommentTok{\# estimation.}
\InTok{In [7]: }\FunctionTok{set\_query!}\NormalTok{(instance, [}\FloatTok{4}\NormalTok{, }\FloatTok{7}\NormalTok{])}
\NormalTok{mmap }\OperatorTok{=} \FunctionTok{MMAPModel}\NormalTok{(instance)}

\OutTok{Out [7]: }\NormalTok{MMAPModel\{}\DataTypeTok{Int64}\NormalTok{, }\DataTypeTok{Array}\NormalTok{\{}\DataTypeTok{Float64}\NormalTok{\}\}}
         \NormalTok{variables}\OperatorTok{:} \FloatTok{4}\NormalTok{, }\FloatTok{7}\NormalTok{ (evidence }\OperatorTok{→} \FloatTok{0}\NormalTok{)}
         \NormalTok{query variables}\OperatorTok{:}\NormalTok{ [[}\FloatTok{1}\NormalTok{, }\FloatTok{2}\NormalTok{, }\FloatTok{6}\NormalTok{, }\FloatTok{5}\NormalTok{, }\FloatTok{3}\NormalTok{, }\FloatTok{8}\NormalTok{]]}
         \NormalTok{contraction time }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{6.022}\NormalTok{, space }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{2.0}\NormalTok{, read}\OperatorTok{{-}}\NormalTok{write }\OperatorTok{=} \FloatTok{2}\OperatorTok{\^{}}\FloatTok{7.033}

\CommentTok{\# Get the most probable configurations for variables 4 and 7.}
\InTok{In [8]: }\FunctionTok{most\_probable\_config}\NormalTok{(mmap)}

\OutTok{Out [8]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{2.8754627318176693}\NormalTok{, [}\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\CommentTok{\# Compute the total log{-}probability of having lung cancer. The results suggest}
\CommentTok{\# that the probability is roughly half.}
\InTok{In [9]: }\FunctionTok{log\_probability}\NormalTok{(mmap, [}\FloatTok{1}\NormalTok{, }\FloatTok{0}\NormalTok{]), }\FunctionTok{log\_probability}\NormalTok{(mmap, [}\FloatTok{0}\NormalTok{, }\FloatTok{0}\NormalTok{])}

\OutTok{Out [9]: }\NormalTok{(}\OperatorTok{{-}}\FloatTok{2.8754627318176693}\NormalTok{, }\OperatorTok{{-}}\FloatTok{2.9206248010671856}\NormalTok{)}
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

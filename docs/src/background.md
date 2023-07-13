# Background

*TensorInference* implements efficient methods to perform Bayesian inference in
*probabilistic graphical models*, such as Bayesian Networks or Markov random
fields.

## Probabilistic graphical models

Probabilistic graphical models (PGMs) capture the mathematical modeling of
reasoning in the presence of uncertainty. Bayesian networks and Markov random
fields are popular types of PGMs. Consider the following Bayesian network known
as the *ASIA network* [^lauritzen1988local]. 

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    % The various elements are conveniently placed using a matrix:
    \matrix[row sep=0.5cm,column sep=0.5cm] {
      % First line
      \node (a) [myvar] {$A$};  &
                                &
                                &
      \node (s) [myvar] {$S$};  &
                               \\
      % Second line
      \node (t) [myvar] {$T$};  &
                                &
      \node (l) [myvar] {$L$};  &
                                &
      \node (b) [myvar] {$B$}; \\
      % Third line
                                &
      \node (e) [myvar] {$E$};  &
                                &
                                &
                               \\
      % Forth line
      \node (x) [myvar] {$X$};  &
                                &
                                &
      \node (d) [myvar] {$D$};  &
                               \\
  };

  \draw [myarrow] (a) edge (t);
  \draw [myarrow] (s) edge (l);
  \draw [myarrow] (s) edge (b);
  \draw [myarrow] (t) edge (e);
  \draw [myarrow] (l) edge (e);
  \draw [myarrow] (e) edge (x);
  \draw [myarrow] (e) edge (d);
  \draw [myarrow] (b) edge (d);
  """,
  options="every node/.style={scale=1.5}",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "asia-network") * "}",
)
save(SVG(joinpath(@__DIR__, "asia-bayesian-network")), tp)
```
![](asia-bayesian-network.svg)

| **Random variable**  | **Meaning**                     |
|        :---:         | :---                            |
|        ``A``         | Recent trip to Asia             |
|        ``T``         | Patient has tuberculosis        |
|        ``S``         | Patient is a smoker             |
|        ``L``         | Patient has lung cancer         |
|        ``B``         | Patient has bronchitis          |
|        ``E``         | Patient hast ``T`` and/or ``L`` |
|        ``X``         | Chest X-Ray is positive         |
|        ``D``         | Patient has dyspnoea            |

The ASIA network corresponds a simplified example from the context of medical
diagnosis that describes the probabilistic relationships between different
random variables corresponding to possible diseases, symptoms, risk factors and
test results. It consists of a graph ``G = (\bm{V},\mathcal{E})`` and a
probability distribution ``P(\bm{V})`` where ``G`` is a directed acyclic graph,
``\bm{V}`` is the set of variables and ``\mathcal{E}`` is the set of edges
connecting the variables. We assume all variables to be discrete. Each variable
``V`` is quantified with a *conditional probability distribution* ``P(V \mid
pa(V))`` where ``pa(V)`` are the parents of ``V``. These conditional probability
distributions together with the graph ``G`` induce a *joint probability
distribution* over ``P(\bm{V})``, given by

```math
P(\bm{V}) = \prod_{V\in\bm{V}} P(V \mid pa(V)).
```


## The inference tasks

Each task is performed with respect to a graphical model, denoted as
``\mathcal{M} = \{\bm{V}, \bm{D}, \bm{\phi}\}``, where:

``\bm{V} = \{ V_1 , V_2 , \dots , V_N \}`` is the set of the model’s variables

``\bm{D} = \{ D_{V_1} , D_{V_2} , \dots , D_{V_N} \}`` is the set of discrete
domains for each variable, and

``\bm{\phi} = \{ \phi_1 , \phi_2 , \dots , \phi_N \}`` is the set of factors
that define the joint probability distribution of the model.

The variable set ``\bm{V}`` can be further partitioned into two subsets: the
evidence variables ``\bm{E}`` and the remaining variables ``\bm{V}^\prime =
\bm{V} \setminus \bm{E}``. Furthermore, within the set ``\bm{V}^\prime``, the
subset ``\bm{Q}`` denotes the query variables. These are the variables for which
we aim to estimate or infer values.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    %\draw[help lines] (0,0) grid (10,-7);

    % mrv: the "node distances" refer to the distance between the edge of a shape
    % to the edge of the other shape. That is why I use "ie_aux" and "tasks_aux"
    % below: to have equal distances between nodes with respect to the center of
    % the shapes.

    % row 1
    \node[myroundbox] (rv) {Random Variables\\$\bm{V}$};
    \node[right=of rv](aux1) {};
    \node[right=of aux1,myroundbox] (jd) {Joint Distribution\\$P(\bm{V})$};
    \node[right=of jd](aux2) {};
    \node[right=of aux2,myroundbox] (e) {Evidence\\$\bm{E=e}$};
    \node[right=of e](aux3) {};
    \node[right=of aux3,myroundbox] (qv) {Query Variables\\$\bm{Q}$};
    % row 2
    \node[below=of aux2,myrectbox] (ie) {Inference Engine};
    \node[below=of aux2] (ie_aux) {};
    % row 3
    \node[below=of ie_aux] (tasks_aux) {};
    \node[left=of tasks_aux,myroundbox] (mar) {MAR};
    \node[left=of mar] (aux4) {};
    \node[left=of aux4,myroundbox] (pr) {PR};
    \node[right=of tasks_aux,myroundbox] (map) {MAP};
    \node[right=of map] (aux5) {};
    \node[right=of aux5,myroundbox] (mmap) {MMAP};
    % row 0
    \node[above=of aux2,yshift=-12mm,text=gray] (in) {\textbf{Input}};
    % row 4
    \node[below=of tasks_aux,yshift=7mm,text=gray] (out) {\textbf{Output}};

    %% edges
    \draw[myarrow] (rv) -- (ie);
    \draw[myarrow] (jd) -- (ie);
    \draw[myarrow] (e)  -- (ie);
    \draw[myarrow] (qv) -- (ie);
    \draw[myarrow] (ie) -- (pr);
    \draw[myarrow] (ie) -- (mar);
    \draw[myarrow] (ie) -- (map);
    \draw[myarrow] (ie) -- (mmap);
  """,
  options="transform shape, scale=1.8",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "the-inference-tasks") * "}",
)
save(SVG("the-inference-tasks"), tp)
```
![](the-inference-tasks.svg)

### Probability of evidence (PR)

Computing the partition function (ie. normalizing constant) or probability of
evidence:

```math
PR(\bm{V}^{\prime} \mid \bm{E}=\bm{e}) = \sum_{V^{\prime} \in \bm{V}^{\prime}} \prod_{\phi \in \bm{\phi}} \phi(V^{\prime},\bm{e})
```

This task involves calculating the probability of the observed evidence, which
can be useful for model comparison or anomaly detection. This involves summing
the joint probability over all possible states of the unobserved variables in
the model, given some observed variables. This is a fundamental task in Bayesian
statistics and is often used as a stepping stone for other types of inference."

### Marginal inference (MAR): 

Computing the marginal probability distribution over all variables given
evidence:

```math
MAR(V_i \mid \bm{E}=\bm{e}) = \frac{ \sum_{V^{\prime\prime} \in \bm{V}^{\prime}
\setminus V_i} \prod_{\phi \in \bm{\phi}} \phi(V^{\prime\prime},\bm{e}) }{
    PR(\bm{V}^{\prime} \mid \bm{E}=\bm{e}) }
```

This task involves computing the marginal probability of a subset of variables,
integrating out the others. In other words, it computes the probability
distribution of some variables of interest regardless of the states of all other
variables. This is useful when we're interested in the probabilities of some
specific variables in the model, but not the entire model.

### Maximum a Posteriori Probability estimation (MAP)

Computing the most likely assignment to all variables given evidence:

```math
MAP(V_i \mid \bm{E}=\bm{e}) = \arg \max_{V^{\prime} \in \bm{V}^{\prime}}
\prod_{\phi \in \bm{\phi}} \phi(V^{\prime},\bm{e})
```

In the MAP task, given some observed variables, the goal is to find the most
probable assignment of values to some subset of the unobserved variables. It
provides the states of variables that maximize the posterior probability given
some observed evidence. This is often used when we want the most likely
explanation or prediction according to the model.

### Marginal Maximum a Posteriori (MMAP)

Computing the most likely assignment to the query variables, ``\bm{Q} \subset
\bm{V}^{\prime}`` after marginalizing out the remaining variables ``\bm{Z} =
\bm{V}^{\prime} \setminus \bm{Q}``, also known as *hidden* or *latent*
variables:

```math
MMAP(V_i \mid \bm{E}=e) = \arg \max_{Q \in \bm{Q}} \sum_{Z \in \bm{Z}}
\prod_{\phi \in \bm{\phi}} \phi(Q, Z, e)
```

This task is essentially a combination of the MAR and MAP tasks. The MMAP task
involves finding the most probable assignment (the MAP estimate) for a subset of
the variables, while marginalizing over (summing out) the remaining variables.
This task is useful when we want to know the most likely state of some
variables, but there's some uncertainty over others that we need to average out.

[^lauritzen1988local]:
    Steffen L Lauritzen and David J Spiegelhalter. Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 50(2):157–194, 1988.

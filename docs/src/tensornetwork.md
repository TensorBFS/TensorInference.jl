# Tensor networks

We now introduce the core ideas of tensor networks, highlighting their
connections with probabilistic graphical models (PGM) to align the terminology
between them.

For our purposes, a tensor is equivalent to the concept of a factor as defined
in the PGM domain, which we detail more formally below.

## What is a tensor?

*Definition*: A tensor $T$ is defined as:
```math
T: \prod_{V \in \bm{V}} \mathcal{D}_{V} \rightarrow \texttt{number}.
```
Here, the function $T$ maps each possible instantiation of the random
variables in its scope $\bm{V}$ to a generic number type. In the context of tensor networks,
a minimum requirement is that the number type is a commutative semiring.
To define a commutative semiring with the addition operation $\oplus$ and the multiplication operation $\odot$ on a set $S$, the following relations must hold for any arbitrary three elements $a, b, c \in S$.
```math
\newcommand{\mymathbb}[1]{\mathbb{#1}}
\begin{align*}
(a \oplus b) \oplus c = a \oplus (b \oplus c) & \hspace{5em}\text{$\triangleright$ commutative monoid $\oplus$ with identity $\mymathbb{0}$}\\
a \oplus \mymathbb{0} = \mymathbb{0} \oplus a = a &\\
a \oplus b = b \oplus a &\\
&\\
(a \odot b) \odot c = a \odot (b \odot c)  &   \hspace{5em}\text{$\triangleright$ commutative monoid $\odot$ with identity $\mymathbb{1}$}\\
a \odot  \mymathbb{1} =  \mymathbb{1} \odot a = a &\\
a \odot b = b \odot a &\\
&\\
a \odot (b\oplus c) = a\odot b \oplus a\odot c  &  \hspace{5em}\text{$\triangleright$ left and right distributive}\\
(a\oplus b) \odot c = a\odot c \oplus b\odot c &\\
&\\
a \odot \mymathbb{0} = \mymathbb{0} \odot a = \mymathbb{0}
\end{align*}
```
Tensors are represented using multidimensional arrays of nonnegative numbers
with labeled dimensions. These labels correspond to the array's indices, which
in turn represent the set of random variables that the tensor is a function
of. Thus, in this context, the terms **label**, **index**, and
**variable** are synonymous and hence used interchangeably.

## What is a tensor network?

We now turn our attention to defining a **tensor network**, a mathematical
object used to represent a multilinear map between tensors. This concept is
widely employed in fields like condensed matter physics
[^Orus2014][^Pfeifer2014], quantum simulation [^Markov2008][^Pan2022], and
even in solving combinatorial optimization problems [^Liu2023]. It's worth
noting that we use a generalized version of the conventional notation, most
commonly known through the
[eisnum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
function, which is commonly used in high-performance computing. Packages that
implement this conventional notation include
- [numpy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)
- [PyTorch](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/einsum)

This approach allows us to represent a broader range of sum-product
multilinear operations between tensors, thus meeting the requirements of the
PGM field.

*Definition*[^Liu2023]: A tensor network is a multilinear map represented by the triple
$\mathcal{N} = (\Lambda, \mathcal{T}, \bm{\sigma}_0)$, where:
-  $\Lambda$ is the set of variables present in the network
    $\mathcal{N}$.
-  $\mathcal{T} = \{ T^{(k)}_{\bm{\sigma}_k} \}_{k=1}^{M}$ is the set of
    input tensors, where each tensor $T^{(k)}_{\bm{\sigma}_k}$ is identified
    by a superscript $(k)$ and has an associated scope $\bm{\sigma}_k$.
-  $\bm{\sigma}_0$ specifies the scope of the output tensor.

More specifically, each tensor $T^{(k)}_{\bm{\sigma}_k} \in \mathcal{T}$ is
labeled by a string $\bm{\sigma}_k \in \Lambda^{r \left(T^{(k)} \right)}$, where
$r \left(T^{(k)} \right)$ is the rank of $T^{(k)}$. The multilinear map, also
known as the `contraction`, applied to this triple is defined as
```math
\texttt{contract}(\Lambda, \mathcal{T}, \bm{\sigma}_0) = \sum_{\bm{\sigma}_{\Lambda
\setminus [\bm{\sigma}_0]}} \prod_{k=1}^{M} T^{(k)}_{\bm{\sigma}_k},
```
Notably, the summation extends over all instantiations of the variables that
are not part of the output tensor.

As an example, consider matrix multiplication, which can be specified as a
tensor network contraction:
```math
  (AB)_{ik} = \texttt{contract}\left(\{i,j,k\}, \{A_{ij}, B_{jk}\}, ik\right),
```
Here, matrices $A$ and $B$ are input tensors labeled by strings $ij, jk \in
\{i, j, k\}^2$. The output tensor is labeled by string $ik$. Summations run
over indices $\Lambda \setminus [ik] = \{j\}$. The contraction corresponds to
```math
  \texttt{contract}\left(\{i,j,k\}, \{A_{ij}, B_{jk}\}, ik\right) = \sum_j
  A_{ij}B_{jk},
```
In the einsum notation commonly used in various programming languages, this is
equivalent to `ij, jk -> ik`.

Diagrammatically, a tensor network can be represented as an *open hypergraph*.
In this diagram, a tensor maps to a vertex, and a variable maps to a
hyperedge. Tensors sharing the same variable are connected by the same
hyperedge for that variable. The diagrammatic representation of matrix
multiplication is:
```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    \matrix[row sep=0.8cm,column sep=0.8cm,ampersand replacement= \& ] {
      \node (1) {};                                               \&
      \node (a) [mytensor] {$A$};                                 \&
      \node (b) [mytensor] {$B$};                                 \&
      \node (2) {};                                               \&
                                                                  \\
    };
    \draw [myedge, color=c01] (1) edge node[below] {$i$} (a);
    \draw [myedge, color=c02] (a) edge node[below] {$j$} (b);
    \draw [myedge, color=c03] (b) edge node[below] {$k$} (2);
  """,
  options="every node/.style={scale=2.0}",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "the-tensor-network") * "}",
)
save(SVG("the-tensor-network1"), tp)
```

```@raw html
<img src="the-tensor-network1.svg"  style="margin-left: auto; margin-right: auto; display:block; width=50%">
```

In this diagram, we use different colors to denote different hyperedges. Hyperedges for
$i$ and $j$ are left open to denote variables in the output string
$\bm{\sigma}_0$. The reason we use hyperedges rather than regular edges will
become clear in the following star contraction example.
```math
  \texttt{contract}(\{i,j,k,l\}, \{A_{il}, B_{jl}, C_{kl}\}, ijk) = \sum_{l}A_{il}
  B_{jl} C_{kl}
```
The equivalent einsum notation employed by many programming languages is `il,
jl, kl -> ijk`.

Since the variable $l$ is shared across all three tensors, a simple graph
can't capture the diagram's complexity. The more appropriate hypergraph
representation is shown below.
```@eval
using TikzPictures

tp = TikzPicture(
  L"""
    \matrix[row sep=0.4cm,column sep=0.4cm,ampersand replacement= \& ] {
                                  \&
                                  \&
      \node[color=c01] (j) {$j$};            \&
                                  \&
                                  \&
                                    \\
                                  \&
                                  \&
      \node (b) [mytensor] {$B$}; \&
                                  \&
                                  \&
                                    \\
      \node[color=c03] (i) {$i$};            \&
      \node (a) [mytensor] {$A$}; \&
      \node[color=c02] (l) {$l$};            \&
      \node (c) [mytensor] {$C$}; \&
      \node[color=c04] (k) {$k$};            \&
                                    \\
    };
    \draw [myedge, color=c01] (j) edge (b);
    \draw [myedge, color=c02] (b) edge (l);
    \draw [myedge, color=c03] (i) edge (a);
    \draw [myedge, color=c02] (a) edge (l);
    \draw [myedge, color=c02] (l) edge (c);
    \draw [myedge, color=c04] (c) edge (k);
  """,
  options="every node/.style={scale=2.0}",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "the-tensor-network") * "}",
)
save(SVG("the-tensor-network2"), tp)
```

```@raw html
<img src="the-tensor-network2.svg"  style="margin-left: auto; margin-right: auto; display:block; width=50%">
```

As a final note, our definition of a tensor network allows for repeated
indices within the same tensor, which translates to self-loops in their
corresponding diagrams.

## Tensor network contraction orders

The performance of a tensor network contraction depends on the order in which
the tensors are contracted. The order of contraction is usually specified by
binary trees, where the leaves are the input tensors and the internal nodes
represent the order of contraction. The root of the tree is the output tensor.

Numerous approaches have been proposed to determine efficient contraction
orderings, which include:
- Greedy algorithms
- Breadth-first search and Dynamic programming [^Pfeifer2014]
- Graph bipartitioning [^Gray2021]
- Local search [^Kalachev2021]

Some of these have been implemented in the
[OMEinsum](https://github.com/under-Peter/OMEinsum.jl) package. Please check
[Performance Tips](@ref) for more details.

## References

[^Orus2014]:
    Orús R. A practical introduction to tensor networks: Matrix product states and projected entangled pair states[J]. Annals of physics, 2014, 349: 117-158.

[^Markov2008]:
    Markov I L, Shi Y. Simulating quantum computation by contracting tensor networks[J]. SIAM Journal on Computing, 2008, 38(3): 963-981.

[^Pfeifer2014]:
    Pfeifer R N C, Haegeman J, Verstraete F. Faster identification of optimal contraction sequences for tensor networks[J]. Physical Review E, 2014, 90(3): 033315.

[^Gray2021]:
    Gray J, Kourtis S. Hyper-optimized tensor network contraction[J]. Quantum, 2021, 5: 410.

[^Kalachev2021]:
    Kalachev G, Panteleev P, Yung M H. Multi-tensor contraction for XEB verification of quantum circuits[J]. arXiv:2108.05665, 2021.

[^Pan2022]:
    Pan F, Chen K, Zhang P. Solving the sampling problem of the sycamore quantum circuits[J]. Physical Review Letters, 2022, 129(9): 090502.

[^Liu2023]:
    Liu J G, Gao X, Cain M, et al. Computing solution space properties of combinatorial optimization problems via generic tensor networks[J]. SIAM Journal on Scientific Computing, 2023, 45(3): A1239-A1270.

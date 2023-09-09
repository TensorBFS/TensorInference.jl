# Tensor networks

We now introduce the core ideas of tensor networks, highlighting their
connections with the probabilistic graphical models (PGM) domain to align the terminology between them.

For our purposes, a **tensor** is equivalent with the concept of a factor
presented above, which we detail more formally below.

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
We now turn our attention to defining a **tensor network**.
Tensor network a mathematical object that can be used to represent a multilinear map between tensors. It is widely used in condensed matter physics [^Orus2014][^Pfeifer2014] and quantum simulation [^Markov2008][^Pan2022]. It is also a powerful tool for solving combinatorial optimization problems [^Liu2023].
It is important to note that we use a generalized version of the conventional
notation, which is also knwon as the [eisnum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) function that widely used in high performance computing.
Packages that implement the conventional notation include
- [numpy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)
- [PyTorch](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/einsum)

This approach allows us to represent a more extensive set of sum-product multilinear operations between tensors, meeting the requirements of the PGM field.

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
$r \left(T^{(k)} \right)$ is the rank of $T^{(k)}$. The multilinear map, or
the `contraction`, applied to this triple is defined as
```math
\texttt{contract}(\Lambda, \mathcal{T}, \bm{\sigma}_0) = \sum_{\bm{\sigma}_{\Lambda
\setminus [\bm{\sigma}_0]}} \prod_{k=1}^{M} T^{(k)}_{\bm{\sigma}_k},
```
Notably, the summation extends over all instantiations of the variables that
are not part of the output tensor.

As an example, the matrix multiplication can be specified as a tensor network
contraction
```math
  (AB)_{ik} = \texttt{contract}\left(\{i,j,k\}, \{A_{ij}, B_{jk}\}, ik\right),
```
where matrices $A$ and $B$ are input tensors labeled by strings $ij, jk \in
\{i, j, k\}^2$. The output tensor is labeled by string $ik$. The
summation runs over indices $\Lambda \setminus [ik] = \{j\}$. The contraction
corresponds to
```math
  \texttt{contract}\left(\{i,j,k\}, \{A_{ij}, B_{jk}\}, ik\right) = \sum_j
  A_{ij}B_{jk},
```
In programming languages, this is equivalent to einsum notation `ij, jk -> ik`.

Diagrammatically, a tensor network can be represented as an *open hypergraph*. In the tensor network diagram, a tensor is mapped to a vertex,
and a variable is mapped to a hyperedge. If and only if tensors share the same variable, we connect
them with the same hyperedge for that variable. The diagrammatic
representation of matrix multiplication is as bellow.
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
""", options="scale=3.8",
    preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "the-tensor-network") * "}",
    )
save(SVG("the-tensor-network1"), tp)
```

```@raw html
<img src="the-tensor-network1.svg"  style="margin-left: auto; margin-right: auto; display:block; width=50%">
```

Here, we use different colors to denote different hyperedges. Hyperedges for
$i$ and $j$ are left open to denote variables in the output string
$\bm{\sigma}_0$. The reason why we should use hyperedges rather than regular edge
will be made clear by the followng star contraction example.
```math
  \texttt{contract}(\{i,j,k,l\}, \{A_{il}, B_{jl}, C_{kl}\}, ijk) = \sum_{l}A_{il}
  B_{jl} C_{kl}
```
In programming languages, this is equivalent to einsum notation `il, jl, kl -> ijk`.

Among the variables, $l$ is shared by all three tensors, hence the diagram can
not be represented as a simple graph. The hypergraph representation is as
below.
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
""", options="",
    preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "the-tensor-network") * "}",
    )
save(SVG("the-tensor-network2"), tp)
```

```@raw html
<img src="the-tensor-network2.svg"  style="margin-left: auto; margin-right: auto; display:block; width=50%">
```

As a final comment, repeated indices in the same tensor is not forbidden in
the definition of a tensor network, hence self-loops are also allowed in a tensor
network diagram.

## Tensor network contraction orders
The performance of a tensor network contraction depends on the order in which
the tensors are contracted. The order of contraction is usually specified by
binary trees, where the leaves are the input tensors and the internal nodes
represent the order of contraction. The root of the tree is the output tensor.

Plenty of algorithms have been proposed to find the optimal contraction order, which includes
- Greedy algorithms
- Breadth-first search and Dynamic programming [^Pfeifer2014]
- Graph bipartitioning [^Gray2021]
- Local search [^Kalachev2021]

Some of them have already been included in the [OMEinsum](https://github.com/under-Peter/OMEinsum.jl) package. Please check [Performance Tips](@ref) for more details.

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
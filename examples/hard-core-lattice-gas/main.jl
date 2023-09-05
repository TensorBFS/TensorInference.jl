# # The hard core lattice gas
# ## Hard-core lattice gas problem
# Hard-core lattice gas refers to a model used in statistical physics to study the behavior of particles on a lattice, where the particles are subject to an exclusion principle known as the "hard-core" interaction that characterized by a blockade radius. Distances between two particles can not be smaller than this radius.
# 
# * Nath T, Rajesh R. Multiple phase transitions in extended hard-core lattice gas models in two dimensions[J]. Physical Review E, 2014, 90(1): 012120.
# * Fernandes H C M, Arenzon J J, Levin Y. Monte Carlo simulations of two-dimensional hard core lattice gases[J]. The Journal of chemical physics, 2007, 126(11).
# 
# Let define a $10 \times 10$ triangular lattice, with unit vectors
# ```math
# \begin{align}
# \vec a &= \left(\begin{matrix}1 \\ 0\end{matrix}\right)\\
# \vec b &= \left(\begin{matrix}\frac{1}{2} \\ \frac{\sqrt{3}}{2}\end{matrix}\right)
# \end{align}
# ```

a, b = (1, 0), (0.5, 0.5*sqrt(3))
Na, Nb = 10, 10
sites = [a .* i .+ b .* j for i=1:Na, j=1:Nb]

# There exists blockade interactions between hard-core particles.
# We connect two lattice sites within blockade radius by an edge.
# Two ends of an edge can not both be occupied by particles.
blockade_radius = 1.1
using GenericTensorNetworks: show_graph, unit_disk_graph
using GenericTensorNetworks.Graphs: edges, nv
graph = unit_disk_graph(vec(sites), blockade_radius)
show_graph(graph; locs=sites, texts=fill("", length(sites)))

# These constraints defines a independent set problem that characterized by the following energy based model.
# Let $G = (V, E)$ be a graph, where $V$ is the set of vertices and $E$ be the set of edges. The energy model for the hard-core lattice gas problem is
# ```math
# E(\mathbf{n}) = -\sum_{i \in V}w_i n_i + \infty \sum_{(i, j) \in E} n_i n_j
# ```
# where $n_i \in \{0, 1\}$ is the number of particles at site $i$, and $w_i$ is the weight associated with it. For unweighted graphs, the weights are uniform.
# The solution space hard-core lattice gas is equivalent to that of an independent set problem. The independent set problem involves finding a set of vertices in a graph such that no two vertices in the set are adjacent (i.e., there is no edge connecting them).
# One can create a tensor network based modeling of an independent set problem with package [`GenericTensorNetworks.jl`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
using GenericTensorNetworks
problem = IndependentSet(graph; optimizer=GreedyMethod());

# There has been a lot of discussions related to solution space properties in the `GenericTensorNetworks` [documentaion page](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/IndependentSet/).
# In this example, we show how to use `TensorInference` to use probabilistic inference for understand the finite temperature properties of this statistic physics model.
# We use [`TensorNetworkModel`](@ref) to convert a combinatorial optimization problem to a probabilistic model.
# Here, we let the inverse temperature be $\beta = 3$.

# ## Probabilistic modeling correlation functions
using TensorInference
β = 3.0
pmodel = TensorNetworkModel(problem, β)

# The partition function of this statistical model can be computed with the [`probability`](@ref) function.
partition_func = probability(pmodel)

# The default return value is a log-rescaled tensor. Use indexing to get the real value.
partition_func[]

# The marginal probabilities can be computed with the [`marginals`](@ref) function, which measures how likely a site is occupied.
mars = marginals(pmodel)
show_graph(graph; locs=sites, vertex_colors=[(b = mars[[i]][2]; (1-b, 1-b, 1-b)) for i in vertices(graph)], texts=fill("", nv(graph)))
# The can see the sites at the corner is more likely to be occupied.
# To obtain two-site correlations, one can set the variables to query marginal probabilities manually.
pmodel2 = TensorNetworkModel(problem, β; mars=[[e.src, e.dst] for e in edges(graph)])
mars = marginals(pmodel2);

# We show the probability that both sites on an edge are not occupied
show_graph(graph; locs=sites, edge_colors=[(b = mars[[e.src, e.dst]][1, 1]; (1-b, 1-b, 1-b)) for e in edges(graph)], texts=fill("", nv(graph)), edge_line_width=5)

# ## The most likely configuration
# The MAP and MMAP can be used to get the most likely configuration given an evidence.
# The relavant function is [`most_probable_config`](@ref).
# If we fix the vertex configuration at one corner to be one, we get the most probably configuration as bellow.
pmodel3 = TensorNetworkModel(problem, β; evidence=Dict(1=>1))
mars = marginals(pmodel3)
logp, config = most_probable_config(pmodel3)

# The log probability is 102. Let us visualize the configuration.
show_graph(graph; locs=sites, vertex_colors=[(1-b, 1-b, 1-b) for b in config], texts=fill("", nv(graph)))
# The number of particles is
sum(config)

# Otherwise, we will get a suboptimal configuration.
pmodel3 = TensorNetworkModel(problem, β; evidence=Dict(1=>0))
logp2, config2 = most_probable_config(pmodel)

# The log probability is 99, which is much smaller.
show_graph(graph; locs=sites, vertex_colors=[(1-b, 1-b, 1-b) for b in config2], texts=fill("", nv(graph)))
# The number of particles is
sum(config2)

## Sampling configurations
# One can ue [`sample`](@ref) to generate samples from hard-core lattice gas at finite temperature.
# The return value is a matrix, with the columns correspond to different samples.
configs = sample(pmodel3, 1000)
sizes = sum(configs; dims=1)
[count(==(i), sizes) for i=0:34]  # counting sizes

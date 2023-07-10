# Performance Tips
## Optimize contraction orders

Let us use the independent set problem on 3-regular graphs as an example.
```julia
julia> using TensorInference, Artifacts, Pkg

julia> Pkg.ensure_artifact_installed("uai2014", pkgdir(TensorInference, "test", "Artifacts.toml"));

julia> function get_instance_filepaths(problem_name::AbstractString, task::AbstractString)
        model_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai")
        evidence_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai.evid")
        solution_filepath = joinpath(artifact"uai2014", task, problem_name * ".uai." * task)
        return model_filepath, evidence_filepath, solution_filepath
    end

julia> model_filepath, evidence_filepath, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")

julia> instance = read_instance(model_filepath; evidence_filepath, solution_filepath)
```

Next, we select the tensor network contraction order optimizer.
```julia
julia> optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
```

Here, we choose the local search based [`TreeSA`](@ref) algorithm, which often finds the smallest time/space complexity and supports slicing.
One can type `?TreeSA` in a Julia REPL for more information about how to configure the hyper-parameters of the [`TreeSA`](@ref) method, 
while the detailed algorithm explanation is in [arXiv: 2108.05665](https://arxiv.org/abs/2108.05665).
Alternative tensor network contraction order optimizers include
* [`GreedyMethod`](@ref) (default, fastest in searching speed but worst in contraction complexity)
* [`KaHyParBipartite`](@ref)
* [`SABipartite`](@ref)

```julia
julia> tn = TensorNetworkModel(instance; optimizer)
```
The returned object `tn` contains a field `code` that specifies the tensor network with optimized contraction order. To check the contraction complexity, please type
```julia
julia> contraction_complexity(problem)
```

The returned object contains log2 values of the number of multiplications, the number elements in the largest tensor during contraction and the number of read-write operations to tensor elements.

```julia
julia> p1 = probability(tn)
```

## Slicing technique

For large scale applications, it is also possible to slice over certain degrees of freedom to reduce the space complexity, i.e.
loop and accumulate over certain degrees of freedom so that one can have a smaller tensor network inside the loop due to the removal of these degrees of freedom.
In the [`TreeSA`](@ref) optimizer, one can set `nslices` to a value larger than zero to turn on this feature.

```julia
julia> tn = TensorNetworkModel(instance; optimizer=TreeSA());

julia> contraction_complexity(tn)
(20.856518235241687, 16.0, 18.88208476145812)
```

As a comparision we slice over 5 degrees of freedom, which can reduce the space complexity by at most 5.
In this application, the slicing achieves the largest possible space complexity reduction 5, while the time and read-write complexity are only increased by less than 1,
i.e. the peak memory usage is reduced by a factor ``32``, while the (theoretical) computing time is increased by at a factor ``< 2``.
```
julia> tn = TensorNetworkModel(instance; optimizer=TreeSA(nslices=5));

julia> timespacereadwrite_complexity(problem)
(21.134967710592804, 11.0, 19.84529401927876)
```

## GEMM for Tropical numbers
No extra effort is required to enjoy the BLAS level speed provided by [`TropicalGEMM`](https://github.com/TensorBFS/TropicalGEMM.jl).
The benchmark in the `TropicalGEMM` repo shows this performance is close to the theoretical optimal value.
Its implementation on GPU is under development in Github repo [`CuTropicalGEMM.jl`](https://github.com/ArrogantGao/CuTropicalGEMM.jl) as a part of [Open Source Promotion Plan summer program](https://summer-ospp.ac.cn/).

## Working with GPUs
To upload the computation to GPU, you just add `using CUDA` before calling the `solve` function, and set the keyword argument `usecuda` to `true`.
```julia
julia> using CUDA
[ Info: OMEinsum loaded the CUDA module successfully

julia> marginals(tn; usecuda = true)
```

Functions support `usecuda` keyword argument includes
* [`probability`](@ref)
* [`log_probability`](@ref)
* [`marginals`](@ref)
* [`most_probable_config`](@ref)

## Benchmarks
Please check our [paper (link to be added)]().

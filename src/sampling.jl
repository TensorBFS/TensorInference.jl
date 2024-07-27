############ Sampling ############
"""
$TYPEDEF

### Fields
$TYPEDFIELDS

The sampled configurations are stored in `samples`, which is a vector of vector.
`labels` is a vector of variable names for labeling configurations.
The `setmask` is an boolean indicator to denote whether the sampling process of a variable is complete.
"""
struct Samples{L} <: AbstractVector{SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
    samples::Matrix{Int}  # size is nvars × nsample
    labels::Vector{L}
    setmask::BitVector
end
function set_eliminated!(samples::Samples, eliminated_variables)
    for var in eliminated_variables
        loc = findfirst(==(var), samples.labels)
        samples.setmask[loc] && error("varaible `$var` is already eliminated.")
        samples.setmask[loc] = true
    end
    return samples
end
Base.getindex(s::Samples, i::Int) = view(s.samples, :, i)
Base.length(s::Samples) = size(s.samples, 2)
Base.size(s::Samples) = (size(s.samples, 2),)
eliminated_variables(samples::Samples) = samples.labels[samples.setmask]
idx4labels(totalset, labels)::Vector{Int} = map(v->findfirst(==(v), totalset), labels)

"""
$(TYPEDSIGNATURES)

The backward process for sampling configurations.

### Arguments
* `code` is the contraction code in the current step,
* `env` is the environment tensor,
* `samples` is the samples generated for eliminated variables,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling!(code::EinCode, @nospecialize(xs::Tuple), @nospecialize(y), @nospecialize(env), samples::Samples, size_dict)
    ixs, iy = getixsv(code), getiyv(code)
    el = setdiff(vcat(ixs...), iy)
    # get probability
    prob_code = optimize_code(EinCode([ixs..., iy], el), size_dict, GreedyMethod(; nrepeat=1))
    probabilities = einsum(prob_code, (xs..., env), size_dict)

    # sample from the probability tensor
    totalset = CartesianIndices((map(x->size_dict[x], el)...,))
    eliminated_locs = idx4labels(samples.labels, el)
    for i=axes(samples.samples, 2)
        config = StatsBase.sample(totalset, Weights(vec(selectdim(probabilities, ndims(probabilities), i))))
        samples.samples[eliminated_locs, i] .= config.I .- 1
    end

    # eliminate the sampled variables
    set_eliminated!(samples, el)
    for l in el
        size_dict[l] = 1
    end
    for sample in sampels
        map(x->eliminate_dimensions!(x, el=>sample), xs)
    end

    # update environment
    for (i, ix) in enumerate(ixs)
    end
    return envs
end

function addbatch(samples::Samples, eliminated_variables)
    uniquelabels = unique!(vcat(ixs..., iy))
    labelmap = Dict(zip(uniquelabels, 1:length(uniquelabels)))
    batchdim = length(labelmap) + 1
    newnewixs = [Int[getindex.(Ref(labelmap), ix)..., batchdim] for ix in newixs]
    newnewiy = Int[getindex.(Ref(labelmap), eliminated_variables)..., batchdim]
    newnewxs = [get_slice(x, dimx, samples.samples[ixloc, :]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
end

# type unstable
function get_slice(x::AbstractArray{T}, slicedim, configs::AbstractMatrix) where T
    outdim = setdiff(1:ndims(x), slicedim)
    res = similar(x, [size(x, d) for d in outdim]..., size(configs, 2))
    return get_slice!(res, x, outdim, slicedim, configs)
end

function get_slice!(res, x::AbstractArray{T}, outdim, slicedim, configs::AbstractMatrix) where T
    xstrides = strides(x)
    @inbounds for ci in CartesianIndices(res)
        idx = 1
        # the output dimension part
        for (dim, k) in zip(outdim, ci.I)
            idx += (k-1) * xstrides[dim]
        end
        # the sliced part
        batchidx = ci.I[end]
        for (dim, k) in zip(slicedim, view(configs, :, batchidx))
            idx += k * xstrides[dim]
        end
        res[ci] = x[idx]
    end
    return res
end

"""
$(TYPEDSIGNATURES)

Generate samples from a tensor network based probabilistic model.
Returns a vector of vector, each element being a configurations defined on `get_vars(tn)`.

### Arguments
* `tn` is the tensor network model.
* `n` is the number of samples to be returned.
"""
function sample(tn::TensorNetworkModel, n::Int; usecuda = false)::Samples
    # generate tropical tensors with its elements being log(p).
    xs = adapt_tensors(tn; usecuda, rescale = false)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{Int, Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(tn.code, xs, size_dict)
    # initialize `y̅` as the initial batch of samples.
    labels = get_vars(tn)
    iy = getiyv(tn.code)
    setmask = falses(length(labels))
    idx = map(l->findfirst(==(l), labels), iy)
    setmask[idx] .= true
    indices = StatsBase.sample(CartesianIndices(size(cache.content)), Weights(normalize!(vec(LinearAlgebra.normalize!(cache.content)))), n)
    configs = zeros(Int, length(labels), n)
    for i=1:n
        configs[idx, i] .= indices[i].I .- 1
    end
    samples = Samples(configs, labels, setmask)
    # back-propagate
    generate_samples(tn.code, cache, samples, size_dict)
    # set evidence variables
    for (k, v) in tn.evidence
        idx = findfirst(==(k), labels)
        samples.samples[idx, :] .= v
    end
    return samples
end

function generate_samples(se::SlicedEinsum, cache::CacheTree{T}, samples, size_dict::Dict) where {T}
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return generate_samples(se.eins, cache, samples, size_dict)
end
function generate_samples(code::NestedEinsum, cache::CacheTree{T}, env::AbstractArray{T}, samples, size_dict::Dict) where {T}
    if !OMEinsum.isleaf(code)
        xs = ntuple(i -> cache.siblings[i].content, length(cache.siblings))
        envs = backward_sampling!(code.eins, xs, cache.content, env, samples, copy(size_dict))
        for (arg, sib, env) in zip(code.args, cache.siblings, envs)
            generate_samples(arg, sib, env, samples, size_dict)
        end
    end
end

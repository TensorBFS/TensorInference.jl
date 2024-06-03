############ Sampling ############
"""
$TYPEDEF

### Fields
$TYPEDFIELDS

The sampled configurations are stored in `samples`, which is a vector of vector.
`labels` is a vector of variable names for labeling configurations.
"""
struct Samples{L} <: AbstractVector{SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
    samples::Matrix{Int}  # size is nvars × nsample
    labels::Vector{L}     # variable names
end
Base.getindex(s::Samples, i::Int) = view(s.samples, :, i)
Base.length(s::Samples) = size(s.samples, 2)
Base.size(s::Samples) = (size(s.samples, 2),)
# update the samples at [var_idx, sampleidx] with `config`
update_sample!(s::Samples, var_idx, sampleidx, config) = s.samples[var_idx, sampleidx] .= config
# get the index for labels
idx4labels(totalset, labels)::Vector{Int} = map(v->findfirst(==(v), totalset), labels)

mutable struct SampleContext{L}
    const samples::Sample{L}         # the samples
    const mask_sampled::BitVector    # boolean mask for sampled variables
    const mask_tosample::BitVector   # boolean mask for variables to be sampled
    env_labels::Vector{L}
    env_tensor::AbstractArray
end
function setmask!(sc::SampleContext, eliminated_variables)
    for var in eliminated_variables
        loc = findfirst(==(var), sc.samples.labels)
        sc.mask_sampled[loc] && error("varaible `$var` is already eliminated.")
        sc.mask_sampled[loc] = true
    end
    return sc
end

"""
$(TYPEDSIGNATURES)

The backward process for sampling configurations.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `sc` is a `SampleContext` object, which contains the samples and masks for sampled variables.
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling!(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), sc::SampleContext, size_dict)
    eliminated_variables = setdiff(vcat(ixs...), iy)
    eliminated_locs = idx4labels(sc.samples.labels, eliminated_variables)
    setmask!(sc, eliminated_variables)

    # the contraction code to get probability
    newixs = map(ix->setdiff(ix, iy), ixs)
    ix_in_sample = map(ix->idx4labels(sc.samples.labels, ix ∩ iy), ixs)
    slice_xs_dim = map(ix->idx4labels(ix, ix ∩ iy), ixs)

    # relabel and compute probabilities
    uniquelabels = unique!(vcat(ixs..., iy))
    labelmap = Dict(zip(uniquelabels, 1:length(uniquelabels)))
    batchdim = length(labelmap) + 1
    newnewixs = [Int[getindex.(Ref(labelmap), ix)..., batchdim] for ix in newixs]
    newnewiy = Int[getindex.(Ref(labelmap), eliminated_variables)..., batchdim]
    newnewxs = [get_slice(x, dimx, sc.samples.samples[ixloc, :]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
    code = DynamicEinCode(newnewixs, newnewiy)
    probabilities = code(newnewxs...)

    totalset = CartesianIndices((map(x->size_dict[x], eliminated_variables)...,))
    for i=1:length(sc.samples)
        config = StatsBase.sample(totalset, safe_weights(vec(selectdim(probabilities, ndims(probabilities), i))))
        # update the samples
        update_sample!(sc.samples, eliminated_locs, i, config.I .- 1)
    end
    return sc
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
function sample(tn::TensorNetworkModel{LT}, n::Int; usecuda = false)::Samples where LT
    # generate tropical tensors with its elements being log(p).
    xs = adapt_tensors(tn; usecuda, rescale = false)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{LT, Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(tn.code, xs, size_dict)
    # get the variable names, initialize the configurations and environment.
    labels = get_vars(tn)
    samples = Samples(zeros(Int, length(labels), n), labels, falses(length(labels)))
    sc = SampleContext(samples,
        falses(length(labels)),  # the sampled variables
        trues(length(labels)),   # the variables to be sampled
        iy,  # the environment labels
        (x=copy(cache.content); fill!(x, one(eltype(x)))) # the environment tensor
    )
    # generate the initial batch of samples on y.
    iy = getiyv(tn.code)
    idx = map(l->findfirst(==(l), labels), iy)
    setmask!(sc, idx)   # mark the output variables as sampled
    indices = StatsBase.sample(CartesianIndices(size(cache.content)), safe_weights(vec(cache.content)), n)
    for i=1:n
        update_sample!(sc.samples, idx, i, indices[i].I .- 1)
    end
    # back-propagate
    sample_back!(tn.code, cache, sc, env, size_dict)
    # set evidence variables
    for (k, v) in tn.evidence
        idx = findfirst(==(k), labels)
        update_sample!(sc.samples, idx, :, v)
    end
    return samples
end
safe_weights(p::AbstractVector{<:Real}) = Weights(p)
function safe_weights(p::AbstractVector{<:Complex})
    @assert all(x -> x ≈ abs(x), p) "Probabilitic weights must be real! Got: $p"
    return Weights(real(p))
end

function sample_back!(se::SlicedEinsum, cache::CacheTree{T}, samples, env, size_dict::Dict) where {T}
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return sample_back!(se.eins, cache, samples, env, size_dict)
end
function sample_back!(code::NestedEinsum, cache::CacheTree{T}, samples, env, size_dict::Dict) where {T}
    if !OMEinsum.isleaf(code)
        xs = ntuple(i -> cache.siblings[i].content, length(cache.siblings))
        backward_sampling!(OMEinsum.getixs(code.eins), xs, OMEinsum.getiy(code.eins), cache.content, samples, size_dict)
        for (arg, sib) in zip(code.args, cache.siblings)
            subenv = ?
            sample_back!(arg, sib, samples, subenv, size_dict)
        end
    end
end

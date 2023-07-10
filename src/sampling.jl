############ Sampling ############
"""
$TYPEDEF

### Fields
$TYPEDFIELDS

The sampled configurations are stored in `samples`, which is a vector of vector.
`labels` is a vector of variable names for labeling configurations.
The `setmask` is an boolean indicator to denote whether the sampling process of a variable is complete.
"""
struct Samples{L}
    samples::Matrix{Int}  # size is nvars × nsample
    labels::Vector{L}
    setmask::BitVector
end
function setmask!(samples::Samples, eliminated_variables)
    for var in eliminated_variables
        loc = findfirst(==(var), samples.labels)
        samples.setmask[loc] && error("varaible `$var` is already eliminated.")
        samples.setmask[loc] = true
    end
    return samples
end

idx4labels(totalset, labels)::Vector{Int} = map(v->findfirst(==(v), totalset), labels)

"""
$(TYPEDSIGNATURES)

The backward process for sampling configurations.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `samples` is the samples generated for eliminated variables,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling!(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), samples::Samples, size_dict)
    eliminated_variables = setdiff(vcat(ixs...), iy)
    eliminated_locs = idx4labels(samples.labels, eliminated_variables)
    setmask!(samples, eliminated_variables)

    # the contraction code to get probability
    newixs = map(ix->setdiff(ix, iy), ixs)
    ix_in_sample = map(ix->idx4labels(samples.labels, ix ∩ iy), ixs)
    slice_xs_dim = map(ix->idx4labels(ix, ix ∩ iy), ixs)

    # relabel and compute probabilities
    uniquelabels = unique!(vcat(ixs..., iy))
    labelmap = Dict(zip(uniquelabels, 1:length(uniquelabels)))
    batchdim = length(labelmap) + 1
    newnewixs = [Int[getindex.(Ref(labelmap), ix)..., batchdim] for ix in newixs]
    newnewiy = Int[getindex.(Ref(labelmap), eliminated_variables)..., batchdim]
    newnewxs = [get_slice(x, dimx, samples.samples[ixloc, :]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
    code = DynamicEinCode(newnewixs, newnewiy)
    probabilities = code(newnewxs...)

    totalset = CartesianIndices((map(x->size_dict[x], eliminated_variables)...,))
    for i=axes(samples.samples, 2)
        config = StatsBase.sample(totalset, Weights(vec(selectdim(probabilities, ndims(probabilities), i))))
        # update the samplesS
        samples.samples[eliminated_locs, i] .= config.I .- 1
    end
    return samples
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
function sample(tn::TensorNetworkModel, n::Int; usecuda = false)::AbstractMatrix{Int}
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
    for (k, v) in tn.fixedvertices
        idx = findfirst(==(k), labels)
        samples.samples[idx, :] .= v
    end
    return samples.samples
end

function generate_samples(code::NestedEinsum, cache::CacheTree{T}, samples, size_dict::Dict) where {T}
    if !OMEinsum.isleaf(code)
        xs = ntuple(i -> cache.siblings[i].content, length(cache.siblings))
        backward_sampling!(OMEinsum.getixs(code.eins), xs, OMEinsum.getiy(code.eins), cache.content, samples, size_dict)
        for (arg, sib) in zip(code.args, cache.siblings)
            generate_samples(arg, sib, samples, size_dict)
        end
    end
end

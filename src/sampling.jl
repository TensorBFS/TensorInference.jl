############ Sampling ############
struct Samples{L}
    samples::Vector{Vector{Int}}
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

idx4labels(totalset, labels) = map(v->findfirst(==(v), totalset), labels)

"""
$(TYPEDSIGNATURES)

The backward rule for tropical einsum.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ysamples` is the samples generated on the output tensor,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling!(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), samples::Samples, size_dict)
    eliminated_variables = setdiff(vcat(ixs...), iy)
    eliminated_locs = idx4labels(samples.labels, eliminated_variables)
    setmask!(samples, eliminated_variables)

    # the contraction code to get probability
    newiy = eliminated_variables
    iy_in_sample = idx4labels(samples.labels, iy)
    slice_y_dim = collect(1:length(iy))
    newixs = map(ix->setdiff(ix, iy), ixs)
    ix_in_sample = map(ix->idx4labels(samples.labels, ix ∩ iy), ixs)
    slice_xs_dim = map(ix->idx4labels(ix, ix ∩ iy), ixs)
    code = DynamicEinCode(newixs, newiy)

    totalset = CartesianIndices((map(x->size_dict[x], eliminated_variables)...,))
    for (i, sample) in enumerate(samples.samples)
        newxs = [get_slice(x, dimx, sample[ixloc]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
        newy = get_element(y, slice_y_dim, sample[iy_in_sample])
        probabilities = einsum(code, (newxs...,), size_dict) / newy
        config = StatsBase.sample(totalset, Weights(vec(probabilities)))
        # update the samples
        samples.samples[i][eliminated_locs] .= config.I .- 1
    end
    return samples
end

# type unstable
function get_slice(x, dim, config)
    asarray(x[[i ∈ dim ? config[findfirst(==(i), dim)]+1 : Colon() for i in 1:ndims(x)]...], x)
end
function get_element(x, dim, config)
    x[[config[findfirst(==(i), dim)]+1 for i in 1:ndims(x)]...]
end

"""
$(TYPEDSIGNATURES)

Sample a tensor network based probabilistic model.
"""
function sample(tn::TensorNetworkModel, n::Int; usecuda = false)::Samples
    # generate tropical tensors with its elements being log(p).
    xs = adapt_tensors(tn; usecuda, rescale = false)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{Int, Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(tn.code, xs, size_dict)
    # initialize `y̅` as the initial batch of samples.
    labels = OMEinsum.uniquelabels(tn.code)
    iy = getiyv(tn.code)
    setmask = falses(length(labels))
    idx = map(l->findfirst(==(l), labels), iy)
    setmask[idx] .= true
    indices = StatsBase.sample(CartesianIndices(size(cache.content)), Weights(normalize!(vec(LinearAlgebra.normalize!(cache.content)))), n)
    configs = map(indices) do ind
        c=zeros(Int, length(labels))
        c[idx] .= ind.I .- 1
        c
    end
    samples = Samples(configs, labels, setmask)
    # back-propagate
    generate_samples(tn.code, cache, samples, size_dict)
    return samples
end

function generate_samples(code::NestedEinsum, cache::CacheTree{T}, samples, size_dict::Dict) where {T}
    if !OMEinsum.isleaf(code)
        xs = ntuple(i -> cache.siblings[i].content, length(cache.siblings))
        backward_sampling!(OMEinsum.getixs(code.eins), xs, OMEinsum.getiy(code.eins), cache.content, samples, size_dict)
        generate_samples.(code.args, cache.siblings, Ref(samples), Ref(size_dict))
    end
end

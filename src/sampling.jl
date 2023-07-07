############ Sampling ############

########### Backward propagating sampling process ##############
function einsum_backward_rule(eins, xs::NTuple{M, AbstractArray{<:Real}} where {M}, y, size_dict, dy::Samples)
    return backward_sampling(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), y, dy, size_dict)
end

struct Samples{L}
    samples::Vector{Vector{Int}}
    labels::Vector{L}
    setmask::Vector{Bool}
end

"""
$(TYPEDSIGNATURES)

The backward rule for tropical einsum.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ysamples` is the samples generated on the output tensor,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), samples::Samples, size_dict)
    idx4label(totalset, labels) = map(v->findfirst(==(v), totalset), labels)
    eliminated_variables = setdiff(vcat(ixs...), iy)
    eliminated_locs = idx4label(samples.labels, eliminated_variables)
    samples.setmask[eliminated_locs] .= true

    # the contraction code to get probability
    newiy = eliminated_variables
    iy_in_sample = idx4labels(samples.labels, iy)
    slice_y_dim = collect(1:length(iy))
    newixs = map(ix->setdiff(ix, iy), ixs)
    ix_in_sample = map(ix->idx4labels(samples.labels, ix ∩ iy), ixs)
    slice_xs_dim = map(ix->idx4label(ix, ix ∩ iy), ixs)
    code = DynamicEinCode(newixs, newiy)

    totalset = CartesianIndices(map(x->size_dict[x], eliminated_variables)...)
    for (i, sample) in enumerate(samples.samples)
        newxs = [get_slice(x, dimx, sample[ixloc]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
        newy = Array(get_slice(y, slice_y_dim, sample[iy_in_sample]))[]
        probabilities = einsum(code, newxs, size_dict) / newy
        config = StatsBase.sample(totalset, weights=StatsBase.Weights(probabilities))
        # update the samples
        samples.samples[i][eliminated_locs] .= config.I
    end
    return xsamples
end

# type unstable
function get_slice(x, dim, config)
    for (d, c) in zip(dim, config)
        x = selectdim(x, d, c)
    end
    return x
end

"""
$(TYPEDSIGNATURES)

Sample a tensor network based probabilistic model.
"""
function sample(tn::TensorNetworkModel; usecuda = false)::AbstractArray{<:Real}
    # generate tropical tensors with its elements being log(p).
    tensors = adapt_tensors(tn; usecuda, rescale = false)
    return tn.code(tensors...)
end

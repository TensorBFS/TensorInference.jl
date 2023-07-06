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
    eliminated_variables = setdiff(vcat(ixs...), iy)
    newiy = eliminated_variables
    newixs = eliminated_variables
    code = DynamicEinCode(newixs, newiy)
    totalset = CartesianIndices(map(x->size_dict[x], eliminated_variables))
    for (i, sample) in enumerate(samples.samples)
        newxs = [get_slice(x, ix, iy=>sample) for (x, ix) in zip(xs, ixs)]
        newy = Array(get_slice(y, iy, iy=>sample))[]
        probabilities = einsum(code, newxs, size_dict) / newy
        config = StatsBase.sample(totalset, weights=StatsBase.Weights(probabilities))
        update_sample!(samples, i, eliminated_variables=>config)
    end
    return xsamples
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

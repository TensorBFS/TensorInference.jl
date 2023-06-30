############ Sampling ############

########### Backward propagating sampling process ##############
function einsum_backward_rule(eins, xs::NTuple{M, AbstractArray{<:Real}} where {M}, y, size_dict, dy::Samples)
    return backward_sampling(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), y, dy, size_dict)
end

"""
$(TYPEDSIGNATURES)

The backward rule for tropical einsum.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ysamples` is the samples generated on the output tensor,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), @nospecialize(ysamples), size_dict)
    xsamples = []
    for i in eachindex(ixs)
        nixs = OMEinsum._insertat(ixs, i, iy)
        nxs  = OMEinsum._insertat(xs, i, y)
        niy  = ixs[i]
        A    = einsum(EinCode(nixs, niy), nxs, size_dict)

        # compute the mask, one of its entry in `A^{-1}` that equal to the corresponding entry in `X` is masked to true.
        j = argmax(xs[i] ./ inv.(A))
        mask = onehot_like(A, j)
        push!(xsamples, mask)
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

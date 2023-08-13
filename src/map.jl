############ Max a posteriori probability (MAP) ############

########### Backward tropical tensor contraction ##############
# This part is copied from [`GenericTensorNetworks`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
function einsum_backward_rule(eins, xs::NTuple{M, AbstractArray{<:Tropical}} where {M}, y, size_dict, dy)
    return backward_tropical(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), y, dy, size_dict)
end

"""
$(TYPEDSIGNATURES)

The backward rule for tropical einsum.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ymask` is the boolean mask for gradients,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_tropical(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), @nospecialize(ymask), size_dict)
    y .= masked_inv.(y, ymask)
    masks = []
    for i in eachindex(ixs)
        nixs = OMEinsum._insertat(ixs, i, iy)
        nxs  = OMEinsum._insertat(xs, i, y)
        niy  = ixs[i]
        A    = einsum(EinCode(nixs, niy), nxs, size_dict)

        # compute the mask, one of its entry in `A^{-1}` that equal to the corresponding entry in `X` is masked to true.
        j = argmax(xs[i] ./ inv.(A))
        mask = onehot_like(A, j)
        push!(masks, mask)
    end
    return masks
end
masked_inv(x, y) = iszero(y) ? zero(x) : inv(x)
function onehot_like(A::AbstractArray, j)
    mask = zero(A)
    mask[j] = one(eltype(mask))
    return mask
end

"""
$(TYPEDSIGNATURES)

Returns the largest log-probability and the most probable configuration.
"""
function most_probable_config(tn::TensorNetworkModel; usecuda = false)::Tuple{Real, Vector}
    expected_mars = [[l] for l in get_vars(tn)]
    @assert tn.mars[1:length(expected_mars)] == expected_mars "To get the the most probable configuration, the leading elements of `tn.vars` must be `$expected_mars`"
    vars = get_vars(tn)
    tensors = map(t -> Tropical.(log.(t)), adapt_tensors(tn; usecuda, rescale = false))
    logp, grads = cost_and_gradient(tn.code, tensors)
    # use Array to convert CuArray to CPU arrays
    return content(Array(logp)[]), map(k -> haskey(tn.evidence, vars[k]) ? tn.evidence[vars[k]] : argmax(grads[k]) - 1, 1:length(vars))
end

"""
$(TYPEDSIGNATURES)

Returns an output array containing largest log-probabilities.
"""
function maximum_logp(tn::TensorNetworkModel; usecuda = false)::AbstractArray{<:Real}
    # generate tropical tensors with its elements being log(p).
    tensors = map(t -> Tropical.(log.(t)), adapt_tensors(tn; usecuda, rescale = false))
    return broadcasted_content(tn.code(tensors...))
end

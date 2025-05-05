############ Max a posteriori probability (MAP) ############

########### Backward tropical tensor contraction ##############
# This part is copied from [`GenericTensorNetworks`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
function OMEinsum.einsum_backward_rule(eins, xs::NTuple{M, AbstractArray{<:Tropical}} where {M}, y, size_dict, dy)
    return backward_tropical!(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), y, dy, size_dict)
end

"""
$(TYPEDSIGNATURES)

The backward rule for tropical einsum.

* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ymask` is the boolean mask for gradients,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_tropical!(ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), @nospecialize(ymask), size_dict)
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
        # zero-out irrelavant indices, otherwise, the result may be unreliable.
        keep_only!(xs[i], j)
        push!(masks, mask)
    end
    return masks
end
function keep_only!(x::AbstractArray{T}, j) where T
    hotvalue = x[j]
    fill!(x, zero(T))
    x[j] = hotvalue
    return x
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
    tensor_indices = check_queryvars(tn, [[v] for v in 1:tn.nvars])
    tensors = map(t -> Tropical.(log.(t)), adapt_tensors(tn; usecuda, rescale = false))
    logp, grads = cost_and_gradient(tn.code, (tensors...,))
    # use Array to convert CuArray to CPU arrays
    return content(Array(logp)[]), map(k -> haskey(tn.evidence, k) ? tn.evidence[k] : argmax(grads[tensor_indices[k]]) - 1, 1:tn.nvars)
end
# check if the queryvars are included in the unity tensors labels, if yes, return the indices of the unity tensors
function check_queryvars(tn::TensorNetworkModel, queryvars::AbstractVector{Vector{Int}})
    ixs = OMEinsum.getixsv(tn.code)
    indices = [findfirst(==(l), ixs[tn.unity_tensors_idx]) for l in queryvars]
    @assert !any(isnothing, indices) "To get the the most probable configuration, the unity tensors labels must include all variables. Query variables: $queryvars, Unity tensors labels: $(ixs[tn.unity_tensors_idx])"
    return tn.unity_tensors_idx[indices]
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

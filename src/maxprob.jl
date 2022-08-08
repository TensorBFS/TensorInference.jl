############ Max a posteriori probability (MAP) ############

########### Backward tropical tensor contraction ##############
# This part is copied from [`GenericTensorNetworks`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
function einsum_backward_rule(eins, xs::NTuple{M,AbstractArray{<:Tropical}} where M, y, size_dict, dy)
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
    for i=1:length(ixs)
        nixs = OMEinsum._insertat(ixs, i, iy)
        nxs  = OMEinsum._insertat( xs, i, y)
        niy = ixs[i]
        A = einsum(EinCode(nixs, niy), nxs, size_dict)

        # compute the mask, one of its entry in `A^{-1}` that equal to the corresponding entry in `X` is masked to true.
        mask = zero(A)
        j = argmax(xs[i] ./ inv.(A))
        mask[j] = one(eltype(mask))

        push!(masks, mask)
    end
    return masks
end
masked_inv(x, y) = iszero(y) ? zero(x) : inv(x)

"""
$(TYPEDSIGNATURES)

Returns the largest log-probability and the most probable configuration.
"""
function most_probable_config(tn::TensorNetworkModeling; usecuda=false)::Tuple{Tropical,Vector}
    vars = get_vars(tn)
    tensors = map(t->Tropical.(log.(t)), generate_tensors(tn; usecuda))
    logp, grads = cost_and_gradient(tn.code, tensors)
    return logp[], map(k->haskey(tn.fixedvertices, vars[k]) ? tn.fixedvertices[vars[k]] : argmax(grads[k]) - 1, 1:length(vars))
end

"""
$(TYPEDSIGNATURES)

Returns an output array containing largest log-probabilities.
"""
function maximum_logp(tn::TensorNetworkModeling; usecuda=false)::AbstractArray{<:Tropical}
    # generate tropical tensors with its elements being log(p).
    tensors = map(t->Tropical.(log.(t)), generate_tensors(tn; usecuda))
    return tn.code(tensors...)
end
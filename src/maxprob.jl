############ Max a posteriori probability (MAP) ############

# generate tropical tensors with its elements being log(p).
function generate_tropical_tensors(gp::TensorNetworksSolver)
    fixedvertices = gp.fixedvertices
    isempty(fixedvertices) && return tensors
    ixs = getixsv(gp.code)
    # `ix` is the vector of labels (or a degree of freedoms) for a tensor,
    # if a label in `ix` is fixed to a value, do the slicing to the tensor it associates to.
    map(gp.tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        Tropical.(log.(t[dims...]))
    end
end

########### Backward tropical tensor contraction ##############
# This part is copied from [`GenericTensorNetworks`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
function einsum_backward_rule(eins, xs::NTuple{M,AbstractArray{<:Tropical}} where M, y, size_dict, dy)
    return backward_tropical(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), y, dy, size_dict)
end

"""
    backward_tropical(ixs, xs, iy, y, ymask, size_dict)

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
        push!(masks, onehotmask(A, xs[i]))
    end
    return masks
end
masked_inv(x, y) = iszero(y) ? zero(x) : inv(x)

# one of the entry in `A` that equal to the corresponding entry in `X` is masked to true.
function onehotmask(A::AbstractArray{T}, X::AbstractArray{T}) where T
    @assert length(A) == length(X)
    mask = zero(A)
    j = argmax(X ./ inv.(A))
    mask[j] = one(T)
    return mask
end

# Returns the log-probability and the configuration.
function most_probable_config(tn::TensorNetworksSolver)
    cards = uniquelabels(tn.code)
    logp, grads = cost_and_gradient(tn.code, generate_tropical_tensors(tn))
    return logp[], map(k->haskey(tn.fixedvertices, cards[k]) ? tn.fixedvertices[cards[k]] : argmax(grads[k]) - 1, 1:length(cards))
end

# Returns probability and the configuration.
function maximum_logp(tn::TensorNetworksSolver)
    return tn.code(generate_tropical_tensors(tn)...)
end
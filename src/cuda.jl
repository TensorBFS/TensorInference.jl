using .CUDA: CuArray

function onehot_like(A::CuArray, j)
    mask = zero(A)
    CUDA.@allowscalar mask[j] = one(eltype(mask))
    return mask
end

# NOTE: this interface should be in OMEinsum
match_arraytype(::Type{<:CuArray{T, N}}, target::AbstractArray{T, N}) where {T, N} = CuArray(target)
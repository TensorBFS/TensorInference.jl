using .CUDA: CuArray

function onehot_like(A::CuArray, j)
    mask = zero(A)
    CUDA.@allowscalar mask[j] = one(eltype(mask))
    return mask
end

# NOTE: this interface should be in OMEinsum
match_arraytype(::Type{<:CuArray{T, N}}, target::AbstractArray{T, N}) where {T, N} = CuArray(target)

function keep_only!(x::CuArray{T}, j) where T
    CUDA.@allowscalar hotvalue = x[j]
    fill!(x, zero(T))
    CUDA.@allowscalar x[j] = hotvalue
    return x
end

using .CUDA: CuArray

function onehot_like(A::CuArray, j)
    mask = zero(A)
    CUDA.@allowscalar mask[j] = one(eltype(mask))
    return mask
end
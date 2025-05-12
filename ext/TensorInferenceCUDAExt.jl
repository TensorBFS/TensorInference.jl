module TensorInferenceCUDAExt
using CUDA: CuArray
import CUDA
import TensorInference: keep_only!, onehot_like, togpu

function onehot_like(A::CuArray, j)
    mask = zero(A)
    CUDA.@allowscalar mask[j] = one(eltype(mask))
    return mask
end

function keep_only!(x::CuArray{T}, j) where T
    CUDA.@allowscalar hotvalue = x[j]
    fill!(x, zero(T))
    CUDA.@allowscalar x[j] = hotvalue
    return x
end

togpu(x::AbstractArray) = CuArray(x)

end
"""
$(TYPEDEF)

    RescaledArray(α, T) -> RescaledArray

An array data type with a log-prefactor, and a l∞-normalized storage, i.e. the maximum element in a tensor is 1.
This tensor type can avoid the potential underflow/overflow of numbers in a tensor network.
The constructor `RescaledArray(α, T)` creates a rescaled array that equal to `exp(α) * T`.
"""
struct RescaledArray{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    log_factor::T
    normalized_value::AT
end
Base.show(io::IO, c::RescaledArray) = print(io, "exp($(c.log_factor)) * $(c.normalized_value)")
Base.show(io::IO, ::MIME"text/plain", c::RescaledArray) = Base.show(io, c)
Base.Array(c::RescaledArray) = rmul!(Array(c.normalized_value), exp(c.log_factor))
Base.copy(c::RescaledArray) = RescaledArray(c.log_factor, copy(c.normalized_value))
Base.getindex(r::RescaledArray, indices...) = map(x->x * exp(r.log_factor), getindex(r.normalized_value, indices...))

"""
$(TYPEDSIGNATURES)

Returns a rescaled array that equivalent to the input tensor.
"""
function rescale_array(tensor::AbstractArray{T})::RescaledArray where {T}
    maxf = maximum(abs, tensor)
    if iszero(maxf)
        @warn("The maximum value of the array to rescale is 0!")
        return RescaledArray(zero(T), tensor)
    end
    return RescaledArray(T(log(maxf)), OMEinsum.asarray(tensor ./ maxf, tensor))
end

for CT in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$CT, @nospecialize(xs::NTuple{N, RescaledArray}), size_dict::Dict) where {N}
        # The following equality holds
        # einsum(code, exp(α) * A, exp(β) * B, ...) = exp(α * β * ...) * einsum(code, A, B, ...)
        # Hence the einsum is performed on the normalized values, and the factors are added later.
        res = einsum(code, getfield.(xs, :normalized_value), size_dict)
        rescaled = rescale_array(res)
        # a new rescaled array, its factor is 
        return RescaledArray(sum(x -> x.log_factor, xs) + rescaled.log_factor, rescaled.normalized_value)
    end
end

Base.size(arr::RescaledArray) = size(arr.normalized_value)
Base.size(arr::RescaledArray, i::Int) = size(arr.normalized_value, i)

match_arraytype(::Type{<:RescaledArray{T, N, AT}}, target::AbstractArray{T, N}) where {T, N, AT} = rescale_array(match_arraytype(AT, target))

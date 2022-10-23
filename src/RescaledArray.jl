# rescaled array: exp(logf) * value
struct RescaledArray{T, N, AT<:AbstractArray{T, N}} <: AbstractArray{T, N}
    logf::T
    value::AT
end
Base.show(io::IO, c::RescaledArray) = print(io, "exp($(c.logf)) * $(c.value)")
Base.show(io::IO, ::MIME"text/plain", c::RescaledArray) = Base.show(io, c)
Base.Array(c::RescaledArray) = rmul!(Array(c.value), exp(c.logf))
Base.copy(c::RescaledArray) = RescaledArray(c.logf, copy(c.value))

function rescale_array(tensor::AbstractArray{T}) where T
    maxf = maximum(tensor)
    if iszero(maxf)
        @warn("The maximum value of the array to rescale is 0!")
        return RescaledArray(zero(T), tensor)
    end
    return RescaledArray(log(maxf), OMEinsum.asarray(tensor ./ maxf, tensor))
end
function rescale_array(tensor::RescaledArray)
    res = rescale_array(tensor.value)
    return RescaledArray(res.logf + tensor.logf, res.value)
end

for CT in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$CT, @nospecialize(xs::NTuple{N,RescaledArray}), size_dict::Dict) where N
        res = einsum(code, getfield.(xs, :value), size_dict)
        return rescale_array(RescaledArray(sum(x->x.logf, xs), res))
    end
end

Base.size(arr::RescaledArray) = size(arr.value)
Base.size(arr::RescaledArray, i::Int) = size(arr.value, i)

match_arraytype(::Type{<:RescaledArray{T, N}}, target::AbstractArray{T, N}) where {T, N} = rescale_array(target)
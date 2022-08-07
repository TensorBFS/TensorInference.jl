"""
$(TYPEDEF)

### Fields
- `vars`
- `vals`

Encodes a discrete function over the set of variables `vars` that maps each
instantiation of `vars` into a nonnegative number in `vals`.
"""
struct Factor{T,N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

"""
$(TYPEDEF)

Probabilistic modeling with a tensor network.

### Fields
* `vars` is the degree of freedoms in the tensor network.
* `code` is the tensor network contraction pattern.
* `tensors` is the tensors fed into the tensor network.
* `fixedvertices` is a dictionary to specifiy degree of freedoms fixed to certain values.
"""
struct TensorNetworkModeling{LT,ET,MT<:AbstractArray}
    vars::Vector{LT}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{LT,Int}
end

"""
$(TYPEDSIGNATURES)
"""
function TensorNetworkModeling(vars::AbstractVector{LT}, factors::Vector{<:Factor{T}}; openvertices=(), fixedvertices=Dict{LT,Int}(), optimizer=GreedyMethod(), simplifier=nothing)::TensorNetworkModeling where {T,LT}
    # The 1st argument of `EinCode` is a vector of vector of labels for specifying the input tensors, 
    # The 2nd argument of `EinCode` is a vector of labels for specifying the output tensor,
    # e.g.
    # `EinCode([[1, 2], [2, 3]], [1, 3])` is the EinCode for matrix multiplication.
    rawcode = EinCode([[[var] for var in vars]..., [[factor.vars...] for factor in factors]...], collect(LT, openvertices))  # labels for vertex tensors (unity tensors) and edge tensors
    tensors = [[ones(T, 2) for _=1:length(vars)]..., getfield.(factors, :vals)...]
    return TensorNetworkModeling(collect(LT, vars), rawcode, tensors; fixedvertices, optimizer, simplifier)
end
"""
$(TYPEDSIGNATURES)
"""
function TensorNetworkModeling(vars::AbstractVector{LT}, rawcode::EinCode, tensors::Vector{<:AbstractArray}; fixedvertices=Dict{LT,Int}(), optimizer=GreedyMethod(), simplifier=nothing)::TensorNetworkModeling where LT
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    code = optimize_code(rawcode, OMEinsum.get_size_dict(getixsv(rawcode), tensors), optimizer, simplifier)
    TensorNetworkModeling(collect(LT, vars), code, tensors, fixedvertices)
end

"""
$(TYPEDSIGNATURES)

Get the variables in this tensor network, they is also known as legs, labels, or degree of freedoms.
"""
get_vars(tn::TensorNetworkModeling)::Vector = tn.vars

chfixedvertices(tn::TensorNetworkModeling, fixedvertices) = TensorNetworkModeling(tn.vars, tn.code, tn.tensors, fixedvertices)

"""
$(TYPEDSIGNATURES)

Evaluate the probability of `config`.
"""
function probability(tn::TensorNetworkModeling, config)::Real
    assign = Dict(zip(get_vars(tn), config .+ 1))
    return mapreduce(x->x[2][getindex.(Ref(assign), x[1])...], *, zip(getixsv(tn.code), tn.tensors))
end

"""
$(TYPEDSIGNATURES)

Contract the tensor network and return a probability array with its rank specified in the contraction code `tn.code`.
The returned array may not be l1-normalized even if the total probability is l1-normalized, because the evidence `tn.fixedvertices` may not be empty.
"""
function probability(tn::TensorNetworkModeling)::AbstractArray
    return tn.code(generate_tensors(tn)...)
end

function OMEinsum.timespacereadwrite_complexity(tn::TensorNetworkModeling)
    tensors = generate_tensors(tn)
    return timespacereadwrite_complexity(tn.code, OMEinsum.get_size_dict(getixsv(tn.code), tensors))
end
OMEinsum.timespace_complexity(tn::TensorNetworkModeling) = timespacereadwrite_complexity(tn)[1:2]
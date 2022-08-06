"""
$(TYPEDEF)

# Fields
- `vars`
- `vals`

Encodes a discrete function over the set of variables `vars` that maps each
instantiation of `vars` into a nonnegative number in `vals`.
"""
struct Factor{T,N}
  vars::NTuple{N,Int64}
  vals::Array{T,N}
end

# code: the tensor network contraction pattern.
# tensors: the tensors fed into the tensor network.
# fixedvertices: the degree of freedoms fixed to a value.
struct TensorNetworksSolver{ET,MT<:AbstractArray}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{Int,Int}
end

function TensorNetworksSolver(factors::Vector{<:Factor}; openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    tensors = getfield.(factors, :vals)
    # The 1st argument of `EinCode` is a vector of vector of labels for specifying the input tensors, 
    # The 2nd argument of `EinCode` is a vector of labels for specifying the output tensor,
    # e.g.
    # `EinCode([[1, 2], [2, 3]], [1, 3])` is the EinCode for matrix multiplication.
    rawcode = EinCode([[factor.vars...] for factor in factors], collect(Int, openvertices))  # labels for edge tensors
    TensorNetworksSolver(rawcode, tensors; fixedvertices, optimizer, simplifier)
end
function TensorNetworksSolver(rawcode::EinCode, tensors::Vector{<:AbstractArray}; fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    code = optimize_code(rawcode, OMEinsum.get_size_dict(getixsv(rawcode), tensors), optimizer, simplifier)
    TensorNetworksSolver(code, tensors, fixedvertices)
end


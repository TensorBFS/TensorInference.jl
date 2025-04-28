"""
$(TYPEDEF)

### Fields
- `vars`
- `vals`

Encodes a discrete function over the set of variables `vars` that maps each
instantiation of `vars` into a nonnegative number in `vals`.
"""
struct Factor{T, N}
    vars::NTuple{N, Int64}
    vals::Array{T, N}
end
summary(f::Factor) = "Factor($(join(f.vars, ", "))), size = $(size(f.vals))"

"""
$(TYPEDEF)

### Fields
* `nvars` is the number of variables,
* `cards` is a vector of cardinalities for variables,
* `factors` is a vector of factors,
"""
struct UAIModel{ET, FT <: Factor{ET}}
    nvars::Int
    cards::Vector{Int}
    factors::Vector{FT}
end

Base.show(io::IO, ::MIME"text/plain", uai::UAIModel) = Base.show(io, uai)
function Base.show(io::IO, uai::UAIModel)
    println(io, "UAIModel(nvars = $(uai.nvars), nfactors = $(length(uai.factors)))")
    println(io, " cards : $(uai.cards)")
    println(io, " factors : ")
    for (k, f) in enumerate(uai.factors)
        print(io, "  $(summary(f))")
        k == length(uai.factors) || println(io)
    end
end

"""
$(TYPEDEF)

Probabilistic modeling with a tensor network.

### Fields
* `nvars` are the number of variables in the tensor network.
* `code` is the tensor network contraction pattern.
* `tensors` are the tensors fed into the tensor network, the leading tensors are unity tensors associated with `unity_tensors_labels`.
* `evidence` is a dictionary used to specify degrees of freedom that are fixed to certain values.
* `unity_tensors_idx` is a vector of indices of the unity tensors in the `tensors` array. Unity tensors are dummy tensors used to obtain the marginal probabilities.
"""
struct TensorNetworkModel{ET, MT <: AbstractArray}
    nvars::Int
    code::ET
    tensors::Vector{MT}
    evidence::Dict{Int, Int}
    unity_tensors_idx::Vector{Int}
end

"""
$TYPEDSIGNATURES

Update the evidence of a tensor network model, without changing the set of observed variables!

### Arguments
- `tnet` is the [`TensorNetworkModel`](@ref) instance.
- `evidence` is the new evidence, the keys must be a subset of existing evidence.
"""
function update_evidence!(tnet::TensorNetworkModel, evidence::Dict)
    for (k, v) in evidence
        haskey(tnet.evidence, k) || error("`update_evidence!` can only update observed variables!")
        tnet.evidence[k] = v
    end
    return tnet
end

function Base.show(io::IO, tn::TensorNetworkModel)
    open = getiyv(tn.code)
    variables = join([string_var(var, open, tn.evidence) for var in get_vars(tn)], ", ")
    tc, sc, rw = contraction_complexity(tn)
    println(io, "$(typeof(tn))")
    println(io, "variables: $variables")
    print_tcscrw(io, tc, sc, rw)
end
Base.show(io::IO, ::MIME"text/plain", tn::TensorNetworkModel) = Base.show(io, tn)

function string_var(var, open, evidence)
    if var ∈ open && haskey(evidence, var)
        "$var (open, fixed to $(evidence[var]))"
    elseif var ∈ open
        "$var (open)"
    elseif haskey(evidence, var)
        "$var (evidence → $(evidence[var]))"
    else
        "$var"
    end
end

function print_tcscrw(io, tc, sc, rw)
    print(io, "contraction time = 2^$(round(tc; digits=3)), space = 2^$(round(sc; digits=3)), read-write = 2^$(round(rw; digits=3))")
end

"""
$(TYPEDSIGNATURES)

### Keyword Arguments
* `openvars` is the list of variables that remains in the output. If it is not empty, the return value will be a nonzero ranked tensor.
* `evidence` is a dictionary of evidences, the values are integers start counting from 0.
* `optimizer` is the tensor network contraction order optimizer, please check the package [`OMEinsumContractionOrders.jl`](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) for available algorithms.
* `simplifier` is some strategies for speeding up the `optimizer`, please refer the same link above.
* `unity_tensors_labels` is a list of labels for the unity tensors. It is all single variables by default, i.e. `[[1], [2], ..., [n]]`. One can also specify multi-variables, which may increase the computational complexity.
"""
function TensorNetworkModel(
    model::UAIModel{ET, FT};
    openvars = (),
    evidence = Dict{Int,Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing,
    unity_tensors_labels = [[i] for i=1:model.nvars]
) where {ET, FT}
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    rawcode = EinCode([unity_tensors_labels..., [[factor.vars...] for factor in model.factors]...], collect(Int, openvars))  # labels for vertex tensors (unity tensors) and edge tensors
    tensors = Array{ET}[[ones(ET, [model.cards[i] for i in lb]...) for lb in unity_tensors_labels]..., [t.vals for t in model.factors]...]
    size_dict = OMEinsum.get_size_dict(getixsv(rawcode), tensors)
    code = optimize_code(rawcode, size_dict, optimizer, simplifier)
    return TensorNetworkModel(model.nvars, code, tensors, evidence, collect(Int, 1:length(unity_tensors_labels)))
end

"""
$(TYPEDSIGNATURES)

Get the variables in this tensor network, they are also known as legs, labels, or degree of freedoms.
"""
get_vars(tn::TensorNetworkModel)::Vector = 1:tn.nvars

"""
$(TYPEDSIGNATURES)

Get the ardinalities of variables in this tensor network.
"""
function get_cards(tn::TensorNetworkModel; fixedisone = false)::Vector
    size_dict = OMEinsum.get_size_dict(getixsv(tn.code), tn.tensors)
    [fixedisone && haskey(tn.evidence, k) ? 1 : size_dict[k] for k in 1:tn.nvars]
end

chevidence(tn::TensorNetworkModel, evidence) = TensorNetworkModel(tn.vars, tn.code, tn.tensors, evidence)

"""
$(TYPEDSIGNATURES)

Evaluate the log probability (or partition function) of `config`.
"""
function log_probability(tn::TensorNetworkModel, config::Union{Dict, AbstractVector})::Real
    assign = config isa AbstractVector ? Dict(zip(get_vars(tn), config)) : config
    return sum(x -> log(x[2][(getindex.(Ref(assign), x[1]) .+ 1)...]), zip(getixsv(tn.code), tn.tensors))
end
"""
$(TYPEDSIGNATURES)

Evaluate the log probability (or partition function).
It is the logged version of [`probability`](@ref), which is less likely to overflow.
"""
function log_probability(tn::TensorNetworkModel; usecuda = false)::AbstractArray
    res = probability(tn; usecuda, rescale=true)
    return asarray(res.log_factor .+ log.(res.normalized_value), res.normalized_value)
end

"""
$(TYPEDSIGNATURES)

Contract the tensor network and return an array of probability of evidence.
Precisely speaking, the return value is the partition function, which may not be l1-normalized.

If the `openvars` of the input tensor networks is zero, the array rank is zero.
Otherwise, the return values corresponds to marginal probabilities.
"""
function probability(tn::TensorNetworkModel; usecuda = false, rescale = true)::AbstractArray
    return tn.code(adapt_tensors(tn; usecuda, rescale)...)
end

"""
    contraction_complexity(tensor_network)

Returns the contraction complexity of a tensor newtork model.
"""
function OMEinsum.contraction_complexity(tn::TensorNetworkModel)
    return contraction_complexity(tn.code, Dict(zip(get_vars(tn), get_cards(tn; fixedisone = true))))
end

# adapt array type with the target array type
match_arraytype(::Type{<:Array{T, N}}, target::AbstractArray{T, N}) where {T, N} = Array(target)

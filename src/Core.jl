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
* `nclique` is the number of cliques,
* `cards` is a vector of cardinalities for variables,
* `factors` is a vector of factors,
"""
struct UAIModel{ET, FT <: Factor{ET}}
    nvars::Int
    nclique::Int
    cards::Vector{Int}
    factors::Vector{FT}
end

Base.show(io::IO, ::MIME"text/plain", uai::UAIModel) = Base.show(io, uai)
function Base.show(io::IO, uai::UAIModel)
    println(io, "UAIModel(nvars = $(uai.nvars), nclique = $(uai.nclique))")
    println(io, " variables :")
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
* `vars` are the degrees of freedom in the tensor network.
* `code` is the tensor network contraction pattern.
* `tensors` are the tensors fed into the tensor network.
* `evidence` is a dictionary used to specify degrees of freedom that are fixed to certain values.
"""
struct TensorNetworkModel{LT, ET, MT <: AbstractArray}
    vars::Vector{LT}
    code::ET
    tensors::Vector{MT}
    evidence::Dict{LT, Int}
end

function Base.show(io::IO, tn::TensorNetworkModel)
    open = getiyv(tn.code)
    variables = join([string_var(var, open, tn.evidence) for var in tn.vars], ", ")
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
"""
function TensorNetworkModel(
    model::UAIModel;
    openvars = (),
    evidence = Dict{Int,Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel
    return TensorNetworkModel(
        1:(model.nvars),
        model.cards,
        model.factors;
        openvars,
        evidence,
        optimizer,
        simplifier
    )
end

"""
$(TYPEDSIGNATURES)
"""
function TensorNetworkModel(
    vars::AbstractVector{LT},
    cards::AbstractVector{Int},
    factors::Vector{<:Factor{T}};
    openvars = (),
    evidence = Dict{LT, Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel where {T, LT}
    # The 1st argument of `EinCode` is a vector of vector of labels for specifying the input tensors, 
    # The 2nd argument of `EinCode` is a vector of labels for specifying the output tensor,
    # e.g.
    # `EinCode([[1, 2], [2, 3]], [1, 3])` is the EinCode for matrix multiplication.
    rawcode = EinCode([[[var] for var in vars]..., [[factor.vars...] for factor in factors]...], collect(LT, openvars))  # labels for vertex tensors (unity tensors) and edge tensors
    tensors = Array{T}[[ones(T, cards[i]) for i in 1:length(vars)]..., [t.vals for t in factors]...]
    return TensorNetworkModel(collect(LT, vars), rawcode, tensors; evidence, optimizer, simplifier)
end

"""
$(TYPEDSIGNATURES)
"""
function TensorNetworkModel(
    vars::AbstractVector{LT},
    rawcode::EinCode,
    tensors::Vector{<:AbstractArray};
    evidence = Dict{LT, Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel where {LT}
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    size_dict = OMEinsum.get_size_dict(getixsv(rawcode), tensors)
    code = optimize_code(rawcode, size_dict, optimizer, simplifier)
    TensorNetworkModel(collect(LT, vars), code, tensors, evidence)
end

"""
$(TYPEDSIGNATURES)

Get the variables in this tensor network, they are also known as legs, labels, or degree of freedoms.
"""
get_vars(tn::TensorNetworkModel)::Vector = tn.vars

"""
$(TYPEDSIGNATURES)

Get the cardinalities of variables in this tensor network.
"""
function get_cards(tn::TensorNetworkModel; fixedisone = false)::Vector
    vars = get_vars(tn)
    [fixedisone && haskey(tn.evidence, vars[k]) ? 1 : length(tn.tensors[k]) for k in 1:length(vars)]
end

chevidence(tn::TensorNetworkModel, evidence) = TensorNetworkModel(tn.vars, tn.code, tn.tensors, evidence)

"""
$(TYPEDSIGNATURES)

Evaluate the log probability of `config`.
"""
function log_probability(tn::TensorNetworkModel, config::Union{Dict, AbstractVector})::Real
    assign = config isa AbstractVector ? Dict(zip(get_vars(tn), config)) : config
    return sum(x -> log(x[2][(getindex.(Ref(assign), x[1]) .+ 1)...]), zip(getixsv(tn.code), tn.tensors))
end

"""
$(TYPEDSIGNATURES)

Contract the tensor network and return a probability array with its rank specified in the contraction code `tn.code`.
The returned array may not be l1-normalized even if the total probability is l1-normalized, because the evidence `tn.evidence` may not be empty.
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

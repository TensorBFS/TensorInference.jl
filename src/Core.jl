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

"""
$(TYPEDEF)

### Fields
* `nvars` is the number of variables,
* `nclique` is the number of cliques,
* `cards` is a vector of cardinalities for variables,
* `factors` is a vector of factors,

* `obsvars` is a vector of observed variables,
* `obsvals` is a vector of observed values,
* `queryvars` is a vector of query variables,
* `reference_solution` is a vector with the reference solution.
"""
struct UAIInstance{ET, FT <: Factor{ET}}
    nvars::Int
    nclique::Int
    cards::Vector{Int}
    factors::Vector{FT}

    obsvars::Vector{Int}
    obsvals::Vector{Int}
    queryvars::Vector{Int}
    reference_solution::Union{Vector{Vector{ET}}, Vector{Int}, Float64}
end

"""
$TYPEDSIGNATURES

Set the evidence of an UAI instance.
"""
function set_evidence!(uai::UAIInstance, pairs::Pair{Int}...)
    empty!(uai.obsvars)
    empty!(uai.obsvals)
    for (var, val) in pairs
        push!(uai.obsvars, var)
        push!(uai.obsvals, val)
    end
    return uai
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
struct TensorNetworkModel{LT, ET, MT <: AbstractArray}
    vars::Vector{LT}
    code::ET
    tensors::Vector{MT}
    fixedvertices::Dict{LT, Int}
end

function Base.show(io::IO, tn::TensorNetworkModel)
    open = getiyv(tn.code)
    variables = join([string_var(var, open, tn.fixedvertices) for var in tn.vars], ", ")
    tc, sc, rw = contraction_complexity(tn)
    println(io, "$(typeof(tn))")
    println(io, "variables: $variables")
    print_tcscrw(io, tc, sc, rw)
end
Base.show(io::IO, ::MIME"text/plain", tn::TensorNetworkModel) = Base.show(io, tn)

function string_var(var, open, fixedvertices)
    if var ∈ open && haskey(fixedvertices, var)
        "$var (open, fixed to $(fixedvertices[var]))"
    elseif var ∈ open
        "$var (open)"
    elseif haskey(fixedvertices, var)
        "$var (evidence → $(fixedvertices[var]))"
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
    instance::UAIInstance;
    openvertices = (),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel
    return TensorNetworkModel(
        1:(instance.nvars),
        instance.cards,
        instance.factors;
        openvertices,
        fixedvertices = Dict(zip(instance.obsvars, instance.obsvals)),
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
    openvertices = (),
    fixedvertices = Dict{LT, Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel where {T, LT}
    # The 1st argument of `EinCode` is a vector of vector of labels for specifying the input tensors, 
    # The 2nd argument of `EinCode` is a vector of labels for specifying the output tensor,
    # e.g.
    # `EinCode([[1, 2], [2, 3]], [1, 3])` is the EinCode for matrix multiplication.
    rawcode = EinCode([[[var] for var in vars]..., [[factor.vars...] for factor in factors]...], collect(LT, openvertices))  # labels for vertex tensors (unity tensors) and edge tensors
    tensors = Array{T}[[ones(T, cards[i]) for i in 1:length(vars)]..., [t.vals for t in factors]...]
    return TensorNetworkModel(collect(LT, vars), rawcode, tensors; fixedvertices, optimizer, simplifier)
end

"""
$(TYPEDSIGNATURES)
"""
function TensorNetworkModel(
    vars::AbstractVector{LT},
    rawcode::EinCode,
    tensors::Vector{<:AbstractArray};
    fixedvertices = Dict{LT, Int}(),
    optimizer = GreedyMethod(),
    simplifier = nothing
)::TensorNetworkModel where {LT}
    # `optimize_code` optimizes the contraction order of a raw tensor network without a contraction order specified.
    # The 1st argument is the contraction pattern to be optimized (without contraction order).
    # The 2nd arugment is the size dictionary, which is a label-integer dictionary.
    # The 3rd and 4th arguments are the optimizer and simplifier that configures which algorithm to use and simplify.
    size_dict = OMEinsum.get_size_dict(getixsv(rawcode), tensors)
    code = optimize_code(rawcode, size_dict, optimizer, simplifier)
    TensorNetworkModel(collect(LT, vars), code, tensors, fixedvertices)
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
    [fixedisone && haskey(tn.fixedvertices, vars[k]) ? 1 : length(tn.tensors[k]) for k in 1:length(vars)]
end

chfixedvertices(tn::TensorNetworkModel, fixedvertices) = TensorNetworkModel(tn.vars, tn.code, tn.tensors, fixedvertices)

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
The returned array may not be l1-normalized even if the total probability is l1-normalized, because the evidence `tn.fixedvertices` may not be empty.
"""
function probability(tn::TensorNetworkModel; usecuda = false, rescale = true)::AbstractArray
    return tn.code(adapt_tensors(tn; usecuda, rescale)...)
end

function OMEinsum.contraction_complexity(tn::TensorNetworkModel)
    return contraction_complexity(tn.code, Dict(zip(get_vars(tn), get_cards(tn; fixedisone = true))))
end

# adapt array type with the target array type
match_arraytype(::Type{<:Array{T, N}}, target::AbstractArray{T, N}) where {T, N} = Array(target)

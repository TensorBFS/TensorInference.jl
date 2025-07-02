"""
$TYPEDEF
    BeliefPropgation(nvars::Int, t2v::AbstractVector{Vector{Int}}, tensors::AbstractVector{AbstractArray{T}}) where T

A belief propagation object.

### Fields
- `t2v::Vector{Vector{Int}}`: a mapping from tensors to variables
- `v2t::Vector{Vector{Int}}`: a mapping from variables to tensors
- `tensors::Vector{AbstractArray{T}}`: the tensors
"""
struct BeliefPropgation{T}
    t2v::Vector{Vector{Int}}           # a mapping from tensors to variables
    v2t::Vector{Vector{Int}}           # a mapping from variables to tensors
    tensors::Vector{AbstractArray{T}}                 # the tensors
end
num_tensors(bp::BeliefPropgation) = length(bp.t2v)
ProblemReductions.num_variables(bp::BeliefPropgation) = length(bp.v2t)

function BeliefPropgation(nvars::Int, t2v::AbstractVector{Vector{Int}}, tensors::AbstractVector{AbstractArray{T}}) where {T}
    # initialize the inverse mapping
    v2t = [Int[] for _ in 1:nvars]
    for (i, edge) in enumerate(t2v)
        for v in edge
            push!(v2t[v], i)
        end
    end
    return BeliefPropgation(t2v, v2t, tensors)
end

"""
$(TYPEDSIGNATURES)

Construct a belief propagation object from a [`UAIModel`](@ref).
"""
function BeliefPropgation(uai::UAIModel{T}) where {T}
    return BeliefPropgation(uai.nvars, [collect(Int, f.vars) for f in uai.factors], AbstractArray{T}[f.vals for f in uai.factors])
end

struct BPState{T, VT <: AbstractVector{T}}
    message_in::Vector{Vector{VT}}  # for each variable, we store the incoming messages
    message_out::Vector{Vector{VT}} # the outgoing messages
end

# message_in -> message_out
function process_message!(bp::BPState; normalize, damping)
    for (ov, iv) in zip(bp.message_out, bp.message_in)
        _process_message!(ov, iv, normalize, damping)
    end
end
function _process_message!(ov::Vector, iv::Vector, normalize::Bool, damping)
    # process the message, TODO: speed up if needed!
    for (i, v) in enumerate(ov)
        w = similar(v)
        fill!(w, one(eltype(v)))  # clear the output vector
        for (j, u) in enumerate(iv)
            j != i && (w .*= u)
        end
        normalize && normalize!(w, 1)
        v .= v .* damping + (1 - damping) * w
    end
end

function collect_message!(bp::BeliefPropgation, state::BPState; normalize::Bool)
    for it in 1:num_tensors(bp)
        out = vectors_on_tensor(state.message_in, bp, it)
        _collect_message!(out, bp.tensors[it], vectors_on_tensor(state.message_out, bp, it))
        normalize && normalize!.(out, 1)
    end
end
# collect the vectors associated with the target tensor
function vectors_on_tensor(messages, bp::BeliefPropgation, it::Int)
    return map(bp.t2v[it]) do v
        # the message goes to the idx-th tensor from variable v
        messages[v][findfirst(==(it), bp.v2t[v])]
    end
end
function _collect_message!(vectors_out::Vector, t::AbstractArray, vectors_in::Vector)
    @assert length(vectors_out) == length(vectors_in) == ndims(t) "dimensions mismatch: $(length(vectors_out)), $(length(vectors_in)), $(ndims(t))"
    # TODO: speed up if needed!
    code = star_code(length(vectors_in))
    cost, gradient = cost_and_gradient(code, (t, vectors_in...))
    for (o, g) in zip(vectors_out, conj.(gradient[2:end]))
        o .= g
    end
    return cost[]
end

# star code: contract a tensor with multiple vectors, one for each dimension
function star_code(n::Int)
    ix1, ixrest = collect(1:n), [[i] for i in 1:n]
    ne = DynamicNestedEinsum([DynamicNestedEinsum{Int}(1), DynamicNestedEinsum{Int}(2)], DynamicEinCode([ix1, ixrest[1]], collect(2:n)))
    for i in 2:n
        ne = DynamicNestedEinsum([ne, DynamicNestedEinsum{Int}(i + 1)], DynamicEinCode([ne.eins.iy, ixrest[i]], collect((i + 1):n)))
    end
    return ne
end

function initial_state(bp::BeliefPropgation{T}) where {T}
    size_dict = OMEinsum.get_size_dict(bp.t2v, bp.tensors)
    edges_vectors = Vector{Vector{T}}[]
    for (i, tids) in enumerate(bp.v2t)
        push!(edges_vectors, [ones(T, size_dict[i]) for _ in 1:length(tids)])
    end
    return BPState(deepcopy(edges_vectors), edges_vectors)
end

"""
$(TYPEDSIGNATURES)

Run the belief propagation algorithm, and return the final state and the information about the convergence.

### Arguments
- `bp::BeliefPropgation`: the belief propagation object

### Keyword Arguments
- `max_iter::Int=100`: the maximum number of iterations
- `tol::Float64=1e-6`: the tolerance for the convergence
- `damping::Float64=0.2`: the damping factor for the message update, updated-message = damping * old-message + (1 - damping) * new-message
"""
function belief_propagate(bp::BeliefPropgation; kwargs...)
    state = initial_state(bp)
    info = belief_propagate!(bp, state; kwargs...)
    return state, info
end
struct BPInfo
    converged::Bool
    iterations::Int
end
function belief_propagate!(bp::BeliefPropgation, state::BPState{T}; max_iter::Int = 100, tol = 1e-6, damping = 0.2) where {T}
    pre_message_in = deepcopy(state.message_in)
    for i in 1:max_iter
        collect_message!(bp, state; normalize = true)
        process_message!(state; normalize = true, damping = damping)
        # check convergence
        if all(iv -> all(it -> isapprox(state.message_in[iv][it], pre_message_in[iv][it], atol = tol), 1:length(bp.v2t[iv])), 1:num_variables(bp))
            return BPInfo(true, i)
        end
        pre_message_in = deepcopy(state.message_in)
    end
    return BPInfo(false, max_iter)
end

# if BP is exact and converged (e.g. tree like), the result should be the same as the tensor network contraction
function contraction_results(state::BPState{T}) where {T}
    return [sum(reduce((x, y) -> x .* y, mi)) for mi in state.message_in]
end

"""
$(TYPEDSIGNATURES)
"""
function marginals(state::BPState{T}) where {T}
    return Dict([v] => normalize!(reduce((x, y) -> x .* y, mi), 1) for (v, mi) in enumerate(state.message_in))
end
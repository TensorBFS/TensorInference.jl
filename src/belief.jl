struct BeliefPropgation{T}
    t2v::Vector{Vector{Int}}           # a mapping from tensors to variables
    v2t::Vector{Vector{Int}}           # a mapping from variables to tensors
    tensors::Vector{AbstractArray{T}}                 # the tensors
end
num_tensors(bp::BeliefPropgation) = length(bp.t2v)
ProblemReductions.num_variables(bp::BeliefPropgation) = length(bp.v2t)

function BeliefPropgation(nvars::Int, t2v::AbstractVector{Vector{Int}}, tensors::AbstractVector{AbstractArray{T}}) where T
    # initialize the inverse mapping
    v2t = [Int[] for _ in 1:nvars]
    for (i, edge) in enumerate(t2v)
        for v in edge
            push!(v2t[v], i)
        end
    end
    return BeliefPropgation(t2v, v2t, tensors)
end
function BeliefPropgation(uai::UAIModel{T}) where T
    return BeliefPropgation(uai.nvars, [collect(Int, f.vars) for f in uai.factors], AbstractArray{T}[f.vals for f in uai.factors])
end

struct BPState{T, VT<:AbstractVector{T}}
    message_in::Vector{Vector{VT}}  # for each variable, we store the incoming messages
    message_out::Vector{Vector{VT}} # the outgoing messages
end

# message_in -> message_out
function process_message!(bp::BPState)
    for (ov, iv) in zip(bp.message_out, bp.message_in)
        _process_message!(ov, iv)
    end
end
function _process_message!(ov::Vector, iv::Vector)
    # process the message, TODO: speed up if needed!
    for (i, v) in enumerate(ov)
        fill!(v, one(eltype(v)))  # clear the output vector
        for (j, u) in enumerate(iv)
            j != i && (v .*= u)
        end
    end
end

function collect_message!(bp::BeliefPropgation, state::BPState)
    for it in 1:num_tensors(bp)
        _collect_message!(vectors_on_tensor(state.message_out, bp, it), bp.tensors[it], vectors_on_tensor(state.message_in, bp, it))
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
    cost, gradient = cost_and_gradient(code, [t, vectors_in...])
    for (o, g) in zip(vectors_out, gradient[2:end])
        o .= g
    end
    return cost
end

# star code: contract a tensor with multiple vectors, one for each dimension
function star_code(n::Int)
    ix1, ixrest = collect(1:n), [[i] for i in 1:n]
    ne = DynamicNestedEinsum([DynamicNestedEinsum{Int}(1), DynamicNestedEinsum{Int}(2)], DynamicEinCode([ix1, ixrest[1]], collect(2:n)))
    for i in 2:n
        ne = DynamicNestedEinsum([ne, DynamicNestedEinsum{Int}(i + 1)], DynamicEinCode([ne.eins.iy, ixrest[i]], collect(i+1:n)))
    end
    return ne
end

function initial_state(bp::BeliefPropgation{T}) where T
    size_dict = OMEinsum.get_size_dict(bp.t2v, bp.tensors)
    edges_vectors = Vector{Vector{T}}[]
    for (i, tids) in enumerate(bp.v2t)
        push!(edges_vectors, [ones(T, size_dict[i]) for _ in 1:length(tids)])
    end
    return BPState(deepcopy(edges_vectors), edges_vectors)
end

# belief propagation, update the tensors on the edges of the tensor network
function belief_propagate(bp::BeliefPropgation; max_iter::Int=100, tol::Float64=1e-6)
    state = initial_state(bp)
    info = belief_propagate!(bp, state; max_iter=max_iter, tol=tol)
    return state, info
end
struct BPInfo
    converged::Bool
    iterations::Int
end
function belief_propagate!(bp::BeliefPropgation, state::BPState{T}; max_iter::Int=100, tol::Float64=1e-6) where T
    for i in 1:max_iter
        process_message!(state)
        collect_message!(bp, state)
        # check convergence
        if all(iv -> all(it -> isapprox(state.message_out[iv][it], state.message_in[iv][it], atol=tol), 1:length(bp.v2t[iv])), 1:num_variables(bp))
            return BPInfo(true, i)
        end
    end
    return BPInfo(false, max_iter)
end

# if BP is exact and converged (e.g. tree like), the result should be the same as the tensor network contraction
function contraction_results(state::BPState{T}) where T
    return [sum(reduce((x, y) -> x .* y, mi)) for mi in state.message_in]
end
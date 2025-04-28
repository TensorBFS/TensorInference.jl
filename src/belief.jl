struct BPState{T, VT<:AbstractVector{T}}
    t2v::Vector{Vector{Int}}           # a mapping from tensors to variables
    v2t::Vector{Vector{Int}}           # a mapping from variables to tensors
    tensors::Vector{AbstractArray{T}}                 # the tensors
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

function collect_message!(bp::BPState)
    for (it, t) in enumerate(bp.t2v)
        _collect_message!(vectors_on_tensor(bp.message_out, bp, it), t, vectors_on_tensor(bp.message_in, bp, it))
    end
end
# collect the vectors associated with the target tensor
function vectors_on_tensor(messages, bp::BPState, it::Int)
    return map(bp.t2v[it]) do v
        # the message goes to the idx-th tensor from variable v
        messages[v][findfirst(==(it), bp.v2t[v])]
    end
end
function _collect_message!(vectors_out::Vector, t::AbstractArray, vectors_in::Vector)
    @assert length(vectors_out) == length(vectors_in) == ndims(t)
    # TODO: speed up if needed!
    code = star_code(length(vectors_in))
    cost, gradient = cost_and_gradient(code, [t, vectors_in...])
    for (o, g) in zip(vectors_out, gradient[2:end])
        o .= g
    end
    return cost
end
function star_code(n::Int)
    ix1, ixrest = collect(1:n), [[i] for i in 1:n]
    ne = DynamicNestedEinsum([DynamicNestedEinsum{Int}(1), DynamicNestedEinsum{Int}(2)], DynamicEinCode([ix1, ixrest[1]], collect(2:n)))
    for i in 2:n
        ne = DynamicNestedEinsum([ne, DynamicNestedEinsum{Int}(i + 1)], DynamicEinCode([ne.eins.iy, ixrest[i]], collect(i+1:n)))
    end
    return ne
end

function BPState(::Type{T}, n::Int, t2v::Vector{Vector{Int}}, size_dict::Dict{Int, Int}) where T
    v2t = [Int[] for _ in 1:n]
    edges_vectors = [Vector{VT}[] for _ in 1:n]
    for (i, edge) in enumerate(t2v)
        for v in edge
            push!(v2t[v], i)
            push!(edges_vectors[i], ones(T, size_dict[v]))
        end
    end
    return BPState(t2v, v2t, edges_vectors)
end

# belief propagation, update the tensors on the edges of the tensor network
function belief_propagation(tn::TensorNetworkModel{T}, bpstate::BPState{T}; max_iter::Int=100, tol::Float64=1e-6) where T
    # collect the messages from the neighbors
    messages = [similar(bpstate.edges_vectors[it]) for it in 1:length(bpstate.t2v)]
    for (it, vs) in enumerate(bpstate.t2v)
        for (iv, v) in enumerate(vs)
            messages[it][iv] = tn.tensors[v]
        end
    end
    # update the tensors on the edges of the tensor network
    for (it, vs) in enumerate(bpstate.t2v)
        # update the tensor
        for (iv, v) in enumerate(vs)
            bpstate.edges_vectors[it][iv] = zeros(T, size_dict[v])
            for (j, w) in enumerate(vs)
                if j != iv
                    bpstate.edges_vectors[it][iv] += messages[j][iv] * messages[j][iv]
                end
            end
        end
    end
end

function belief_propagation(tn::TensorNetworkModel{T}) where T
    return belief_propagation(tn, BPState(T, OMEinsum.get_ixsv(tn.code), tn.size_dict))
end

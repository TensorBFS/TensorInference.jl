struct BPState{T, VT<:AbstractVector{T}}
    t2v::Vector{Vector{Int}}           # a mapping from tensors to variables
    v2t::Vector{Vector{Int}}           # a mapping from variables to tensors
    edges_vectors::Vector{Vector{VT}}  # each tensor is associated with a vector of vectors, one for each neighbor
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

function tensor_product()
end

function belief_propagation(tn::TensorNetworkModel{T}) where T
    return belief_propagation(tn, BPState(T, OMEinsum.get_ixsv(tn.code), tn.size_dict))
end

bitmask_type(nbits::Int) = LongLongUInt{max(1, cld(nbits, 64))}
@inline mask_indices(mask::T, nbits::Int) where {T<:Integer} = (i for i in 1:nbits if readbit(mask, i) == 1)

abstract type LoopSeriesTruncation end
struct XORLoopSum <: LoopSeriesTruncation; k::Int; end
struct UnionLoopSum <: LoopSeriesTruncation; k::Int; end
struct Degree <: LoopSeriesTruncation; max::Int; end
struct Cyclomatic <: LoopSeriesTruncation; max::Int; end

function edge_list_index(g::SimpleGraph)
    eds = [(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(g)]
    sort!(eds)
    idx = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(eds)
        idx[(u, v)] = i
        idx[(v, u)] = i
    end
    return eds, idx
end

function cycle_mask(cycle, idx, ::Type{T}) where {T<:Integer}
    n = length(cycle); n == 0 && return zero(T)
    mask = zero(T)
    @inbounds for i in 1:n
        u = cycle[i]; v = cycle[i == n ? 1 : i + 1]
        mask |= bmask(T, idx[(u, v)])
    end
    return mask
end

function _generalized_loops(eds, nverts; max_edges, max_order)
    m = length(eds); EdgeMask = bitmask_type(m); loops = EdgeMask[]
    (m == 0 || max_edges <= 2) && return loops
    max_edges = min(max_edges, m); max_order != typemax(Int) && (max_edges = min(max_edges, nverts - 1 + max_order))
    us = first.(eds); vs = last.(eds)
    by_vertex = [Int[] for _ in 1:nverts]
    for i in 1:m; push!(by_vertex[us[i]], i); push!(by_vertex[vs[i]], i); end
    edge_adj = [Int[] for _ in 1:m]
    for inc in by_vertex; for i in inc, j in inc; i == j || push!(edge_adj[i], j); end; end
    for i in 1:m; sort!(edge_adj[i]); unique!(edge_adj[i]); end
    deg = zeros(Int, nverts)
    function add_edge!(idx, vcount, d1)
        u = us[idx]; v = vs[idx]
        deg[u] == 0 && (vcount += 1); deg[v] == 0 && (vcount += 1)
        deg[u] == 1 && (d1 -= 1); deg[v] == 1 && (d1 -= 1)
        deg[u] += 1; deg[v] += 1
        deg[u] == 1 && (d1 += 1); deg[v] == 1 && (d1 += 1)
        return vcount, d1
    end
    remove_edge!(idx) = (deg[us[idx]] -= 1; deg[vs[idx]] -= 1)
    function backtrack(root, edge_mask, edge_count, vcount, d1, cand, cand_mask, excluded)
        cyclo = edge_count - vcount + 1; cyclo > max_order && return
        edge_count > 0 && d1 == 0 && cyclo <= max_order && push!(loops, edge_mask)
        edge_count == max_edges && return
        prefix_mask = zero(EdgeMask)
        for pos in 1:length(cand)
            idx = cand[pos]
            new_cand = cand[pos+1:end]; new_cand_mask = cand_mask & ~prefix_mask & ~bmask(EdgeMask, idx)
            new_excluded = excluded | prefix_mask | bmask(EdgeMask, idx)
            for nb in edge_adj[idx]
                nb <= root && continue
                readbit(edge_mask, nb) == 1 && continue
                readbit(new_excluded, nb) == 1 && continue
                readbit(new_cand_mask, nb) == 1 && continue
                push!(new_cand, nb); new_cand_mask |= bmask(EdgeMask, nb)
            end
            vcount2, d12 = add_edge!(idx, vcount, d1)
            backtrack(root, edge_mask | bmask(EdgeMask, idx), edge_count + 1, vcount2, d12, new_cand, new_cand_mask, new_excluded)
            remove_edge!(idx); prefix_mask |= bmask(EdgeMask, idx)
        end
    end
    for root in 1:m
        edge_mask = bmask(EdgeMask, root)
        vcount, d1 = add_edge!(root, 0, 0)
        cand = [nb for nb in edge_adj[root] if nb > root]
        cand_mask = isempty(cand) ? zero(EdgeMask) : bmask(EdgeMask, cand)
        backtrack(root, edge_mask, 1, vcount, d1, cand, cand_mask, zero(EdgeMask))
        remove_edge!(root)
    end
    sort!(loops; by = count_ones); return loops
end

function _loop_series_cycles(g, k, op; connected::Bool = false, check_connected::Bool = false)
    eds, idx = edge_list_index(g)
    EdgeMask = bitmask_type(length(eds))
    cycles = [cycle_mask(c, idx, EdgeMask) for c in cycle_basis(g)]
    k = min(k, length(cycles))
    k <= 0 && return (edges = eds, loops = eltype(cycles)[])
    if !connected
        results = Set{eltype(cycles)}()
        function combine_backtrack(start, depth, acc)
            depth > 0 && push!(results, acc)
            depth == k && return
            for i in start:length(cycles)
                combine_backtrack(i + 1, depth + 1, op(acc, cycles[i]))
            end
        end
        combine_backtrack(1, 0, zero(eltype(cycles)))
        delete!(results, zero(eltype(cycles)))
        return (edges = eds, loops = collect(results))
    end
    ncycles = length(cycles); nverts = nv(g); nedges = length(eds)
    VMask = bitmask_type(nverts); CycleMask = bitmask_type(ncycles)
    vmasks = Vector{VMask}(undef, ncycles)
    for i in 1:ncycles
        vmask = zero(VMask)
        for edge_idx in mask_indices(cycles[i], length(eds)); u, v = eds[edge_idx]; vmask |= bmask(VMask, u, v); end
        vmasks[i] = vmask
    end
    adjmask = fill(zero(CycleMask), ncycles)
    for i in 1:ncycles-1
        for j in i+1:ncycles
            iszero(vmasks[i] & vmasks[j]) && continue
            adjmask[i] |= bmask(CycleMask, j); adjmask[j] |= bmask(CycleMask, i)
        end
    end
    us = first.(eds); vs = last.(eds)
    incident = [Int[] for _ in 1:nverts]
    for i in 1:nedges; push!(incident[us[i]], i); push!(incident[vs[i]], i); end
    used = zeros(Int, nverts); seen = zeros(Int, nverts)
    used_mark = 0; seen_mark = 0; used_nodes = Int[]; stack = Int[]
    results = Set{eltype(cycles)}()
    function connected_mask(mask)
        iszero(mask) && return false
        used_mark += 1; start = 0; empty!(used_nodes)
        for idx in mask_indices(mask, nedges)
            u = us[idx]; v = vs[idx]
            if used[u] != used_mark; used[u] = used_mark; push!(used_nodes, u); start == 0 && (start = u); end
            if used[v] != used_mark; used[v] = used_mark; push!(used_nodes, v); start == 0 && (start = v); end
        end
        start == 0 && return false
        seen_mark += 1; empty!(stack); push!(stack, start); seen[start] = seen_mark
        while !isempty(stack)
            u = pop!(stack)
            for e in incident[u]
                readbit(mask, e) == 1 || continue
                v = us[e] == u ? vs[e] : us[e]
                seen[v] == seen_mark && continue
                seen[v] = seen_mark; push!(stack, v)
            end
        end
        for v in used_nodes; seen[v] != seen_mark && return false; end
        return true
    end
    function backtrack(root_lower, sub_cycles, acc_edges, cand_mask, depth)
        depth > 0 && (!check_connected || depth == 1 || connected_mask(acc_edges)) && push!(results, acc_edges)
        depth == k && return
        prefix_mask = zero(CycleMask)
        for v in mask_indices(cand_mask, ncycles)
            vbit = bmask(CycleMask, v)
            new_sub = sub_cycles | vbit; new_acc = op(acc_edges, cycles[v])
            new_cand = cand_mask & ~prefix_mask & ~vbit
            new_cand |= adjmask[v] & ~root_lower & ~new_sub & ~new_cand & ~prefix_mask
            backtrack(root_lower, new_sub, new_acc, new_cand, depth + 1)
            prefix_mask |= vbit
        end
    end
    lower_mask = zero(CycleMask)
    for root in 1:ncycles
        rootbit = bmask(CycleMask, root); lower_mask |= rootbit
        backtrack(lower_mask, rootbit, cycles[root], adjmask[root] & ~lower_mask, 1)
    end
    return (edges = eds, loops = collect(results))
end

loop_series(g::SimpleGraph, trunc::XORLoopSum) = _loop_series_cycles(g, trunc.k, ⊻; connected = true, check_connected = true)
loop_series(g::SimpleGraph, trunc::UnionLoopSum) = _loop_series_cycles(g, trunc.k, |; connected = true)
loop_series(g::SimpleGraph, trunc::Degree) = begin
    corenum = core_number(g)
    eds, _ = edge_list_index(g)
    eds = [e for e in eds if corenum[e[1]] >= 2 && corenum[e[2]] >= 2]
    return (edges = eds, loops = _generalized_loops(eds, nv(g); max_edges = trunc.max, max_order = typemax(Int)))
end
loop_series(g::SimpleGraph, trunc::Cyclomatic) = begin
    if trunc.max == 1
        eds, idx = edge_list_index(g)
        T = bitmask_type(length(eds))
        masks = unique!(collect(cycle_mask(c, idx, T) for c in simplecycles(DiGraph(g)) if length(c) >= 3))
        sort!(masks; by = count_ones)
        return (edges = eds, loops = masks)
    end
    corenum = core_number(g)
    eds, _ = edge_list_index(g)
    eds = [e for e in eds if corenum[e[1]] >= 2 && corenum[e[2]] >= 2]
    return (edges = eds, loops = _generalized_loops(eds, nv(g); max_edges = typemax(Int), max_order = trunc.max))
end

function tensor_connectivity_graph(bp::BeliefPropgation)
    g = SimpleGraph(num_tensors(bp))
    edge_to_var = Dict{Tuple{Int, Int}, Int}()
    for v in 1:num_variables(bp)
        tids = bp.v2t[v]; length(tids) == 2 || continue
        a = min(tids[1], tids[2])
        b = max(tids[1], tids[2])
        haskey(edge_to_var, (a, b)) && throw(ArgumentError("multiple variables between tensors $a and $b"))
        edge_to_var[(a, b)] = v; add_edge!(g, a, b)
    end
    return g, edge_to_var
end

function _loops_from_edge_masks(bp, edges, masks, edge_to_var)
    nvars = num_variables(bp); nt = num_tensors(bp)
    EdgeMask = bitmask_type(nvars); TensorMask = bitmask_type(nt)
    loops = NamedTuple{(:edges, :tensors), Tuple{EdgeMask, TensorMask}}[]
    for mask in masks
        iszero(mask) && continue
        edge_mask = zero(EdgeMask); tensor_mask = zero(TensorMask)
        for edge_idx in mask_indices(mask, length(edges))
            u, v = edges[edge_idx]
            key = u < v ? (u, v) : (v, u)
            var = edge_to_var[key]
            edge_mask |= bmask(EdgeMask, var); tensor_mask |= bmask(TensorMask, u, v)
        end
        push!(loops, (edges = edge_mask, tensors = tensor_mask))
    end
    return loops
end

function loop_basis(bp::BeliefPropgation)
    g, edge_to_var = tensor_connectivity_graph(bp)
    eds, idx = edge_list_index(g)
    EdgeMask = bitmask_type(length(eds))
    cycles = [cycle_mask(c, idx, EdgeMask) for c in cycle_basis(g)]
    return _loops_from_edge_masks(bp, eds, cycles, edge_to_var)
end

function loop_series(bp::BeliefPropgation, trunc::LoopSeriesTruncation)
    g, edge_to_var = tensor_connectivity_graph(bp)
    series = loop_series(g, trunc)
    return _loops_from_edge_masks(bp, series.edges, series.loops, edge_to_var)
end

struct LoopSeriesCache{T, VT <: AbstractVector{T}}; message_in_norm::Vector{Vector{VT}}; complement_proj::Vector{Union{Nothing, Matrix{T}}}; reduced_cache::Vector{Dict{Any, Any}}; plan_cache::Dict{Any, Any}; end

function LoopSeriesCache(bp::BeliefPropgation, state::BPState{T}) where {T}
    nvars = num_variables(bp); nt = num_tensors(bp)
    msg_norm = Vector{Vector{typeof(state.message_in[1][1])}}(undef, nvars)
    c_mats = Vector{Union{Nothing, Matrix{T}}}(undef, nvars)
    for v in 1:nvars
        msgs = state.message_in[v]; msg_norm[v] = [copy(m) for m in msgs]
        if length(msgs) == 2
            m1, m2 = msgs; s = dot(m2, m1)
            iszero(s) && throw(ArgumentError("edge $v has zero message overlap"))
            scale1 = inv(sqrt(s)); scale2 = inv(sqrt(conj(s)))
            msg_norm[v][1] = m1 .* scale1; msg_norm[v][2] = m2 .* scale2
            d = length(m1); P = msg_norm[v][2] * msg_norm[v][1]'
            c_mats[v] = Matrix{T}(I, d, d) - P
        else
            c_mats[v] = nothing
        end
    end
    return LoopSeriesCache(msg_norm, c_mats, [Dict{Any, Any}() for _ in 1:nt], Dict{Any, Any}())
end

function _message_to_tensor(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache, v::Int, t::Int)
    tids = bp.v2t[v]
    if length(tids) == 2
        idx = findfirst(==(t), tids)
        idx === nothing && throw(ArgumentError("tensor $t not attached to variable $v"))
        return cache.message_in_norm[v][idx == 1 ? 2 : 1]
    elseif length(tids) == 1
        return state.message_out[v][1]
    else
        throw(ArgumentError("loop corrections require variables of degree 1 or 2; variable $v has degree $(length(tids))"))
    end
end

function _reduced_tensor(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache, loop_edges, t::Int)
    vars = bp.t2v[t]
    local_mask_type = bitmask_type(length(vars)); local_mask = zero(local_mask_type)
    ixs = [vars]; tensors = Any[bp.tensors[t]]; keep = Int[]
    for (i, v) in enumerate(vars)
        if readbit(loop_edges, v) == 1
            push!(keep, v); local_mask |= bmask(local_mask_type, i)
        else
            msg = _message_to_tensor(bp, state, cache, v, t)
            push!(ixs, [v]); push!(tensors, msg)
        end
    end
    cache_t = cache.reduced_cache[t]
    haskey(cache_t, local_mask) && return cache_t[local_mask], keep
    reduced = EinCode(ixs, keep)(tensors...)
    cache_t[local_mask] = reduced
    return reduced, keep
end

function loop_weight(bp::BeliefPropgation, state::BPState, loop; optimizer = nothing, cache = nothing)
    cache = cache === nothing ? LoopSeriesCache(bp, state) : cache
    nvars = num_variables(bp); nt = num_tensors(bp)
    tensors = Any[]; labels = Vector{Vector{Int}}()
    sizehint!(tensors, count_ones(loop.tensors) + count_ones(loop.edges)); sizehint!(labels, count_ones(loop.tensors) + count_ones(loop.edges))
    for t in mask_indices(loop.tensors, nt)
        reduced, keep_vars = _reduced_tensor(bp, state, cache, loop.edges, t)
        push!(tensors, reduced); push!(labels, [t == bp.v2t[v][1] ? v : v + nvars for v in keep_vars])
    end
    for v in mask_indices(loop.edges, nvars)
        C = cache.complement_proj[v]; C === nothing && throw(ArgumentError("loop edge $v is not degree-2"))
        push!(tensors, C); push!(labels, [v, v + nvars])
    end
    code = EinCode(labels, Int[])
    if optimizer !== nothing
        label_map = Dict{Int, Int}(); dims = Int[]
        canon_labels = Vector{Vector{Int}}(undef, length(labels))
        for (i, labs) in enumerate(labels)
            clabs = Vector{Int}(undef, length(labs))
            for (j, lab) in enumerate(labs)
                cid = get!(label_map, lab) do
                    push!(dims, size(tensors[i], j)); length(dims)
                end
                clabs[j] = cid
            end
            canon_labels[i] = clabs
        end
        key = (Tuple(map(Tuple, canon_labels)), Tuple(dims))
        code = get!(cache.plan_cache, key) do
            size_dict = OMEinsum.get_size_dict(canon_labels, tensors)
            optimize_code(EinCode(canon_labels, Int[]), size_dict, optimizer)
        end
    end
    return code(tensors...)[]
end

function _bp_vacuum_factors(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache)
    empty_loop = zero(bitmask_type(num_variables(bp)))
    return map(1:num_tensors(bp)) do t
        reduced, _ = _reduced_tensor(bp, state, cache, empty_loop, t)
        reduced[]
    end
end

function bp_vacuum_weight(bp::BeliefPropgation, state::BPState; cache = nothing)
    cache = cache === nothing ? LoopSeriesCache(bp, state) : cache
    return prod(_bp_vacuum_factors(bp, state, cache))
end

function loop_corrections(bp::BeliefPropgation, state::BPState; loops, n_edges_trunc::Int = typemax(Int), n_loops_trunc::Int = 1, optimizer = nothing)
    cache = LoopSeriesCache(bp, state)
    vacuums = _bp_vacuum_factors(bp, state, cache)
    bp_weight = prod(vacuums)
    if isempty(loops) || n_edges_trunc <= 0 || n_loops_trunc <= 0
        return (bp_weight = bp_weight, correction = zero(bp_weight), value = bp_weight, loop_weights = typeof(bp_weight)[])
    end
    edge_counts = count_ones.(getfield.(loops, :edges))
    keep = findall(edge_counts .<= n_edges_trunc)
    isempty(keep) && return (bp_weight = bp_weight, correction = zero(bp_weight), value = bp_weight, loop_weights = typeof(bp_weight)[])
    loops = loops[keep]; edge_counts = edge_counts[keep]
    loop_tensors = getfield.(loops, :tensors)
    loop_weights = Vector{typeof(bp_weight)}(undef, length(loops))
    for i in eachindex(loops)
        raw = loop_weight(bp, state, loops[i]; optimizer, cache)
        vac = prod(vacuums[t] for t in mask_indices(loop_tensors[i], num_tensors(bp)))
        iszero(vac) && throw(ArgumentError("loop vacuum factor is zero"))
        loop_weights[i] = raw / vac
    end
    total = zero(bp_weight)
    TensorMask = typeof(loop_tensors[1])
    function backtrack(start, depth, used_tensors, weight_prod, edge_total)
        depth > 0 && (total += weight_prod)
        depth == n_loops_trunc && return
        for i in start:length(loop_weights)
            iszero(loop_tensors[i] & used_tensors) || continue
            new_edge_total = edge_total + edge_counts[i]
            new_edge_total > n_edges_trunc && continue
            backtrack(i + 1, depth + 1, used_tensors | loop_tensors[i], weight_prod * loop_weights[i], new_edge_total)
        end
    end
    backtrack(1, 0, zero(TensorMask), one(bp_weight), 0)
    correction = bp_weight * total
    return (bp_weight = bp_weight, correction = correction, value = bp_weight + correction, loop_weights = loop_weights)
end

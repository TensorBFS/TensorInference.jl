using TensorInference, Test, LinearAlgebra, Graphs, BitBasis, Random

function gf2_basis(masks::Vector{T}, nbits::Int) where {T<:Integer}
    basis = T[]
    pivots = Int[]
    for mask in masks
        vec = mask
        for (b, p) in zip(basis, pivots)
            readbit(vec, p) == 0 && continue
            vec = vec ⊻ b
        end
        iszero(vec) && continue
        pivot = 0
        for i in nbits:-1:1
            if readbit(vec, i) == 1
                pivot = i
                break
            end
        end
        insert_at = findfirst(x -> x < pivot, pivots)
        if insert_at === nothing
            push!(basis, vec)
            push!(pivots, pivot)
        else
            insert!(basis, insert_at, vec)
            insert!(pivots, insert_at, pivot)
        end
    end
    return basis, pivots
end

function gf2_rank(masks::Vector{T}, nbits::Int) where {T<:Integer}
    basis, _ = gf2_basis(masks, nbits)
    return length(basis)
end

function reduce_with_basis(mask::T, basis::Vector{T}, pivots::Vector{Int}) where {T<:Integer}
    vec = mask
    for (b, p) in zip(basis, pivots)
        readbit(vec, p) == 0 && continue
        vec = vec ⊻ b
    end
    return vec
end

function cycle_mask(cycle::Vector{Int}, edge_index, ::Type{T}) where {T<:Integer}
    n = length(cycle)
    n == 0 && return zero(T)
    mask = zero(T)
    for i in 1:n
        u = cycle[i]
        v = cycle[i == n ? 1 : i + 1]
        idx = edge_index[(u, v)]
        mask = mask | bmask(T, idx)
    end
    return mask
end

@testset "cycle basis on Petersen graph" begin
    g = Graphs.SimpleGraphs.smallgraph(:petersen)
    eds = [(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(g)]
    sort!(eds)
    edge_index = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(eds)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    end
    basis_cycles = [cycle_mask(c, edge_index, TensorInference.bitmask_type(length(eds))) for c in cycle_basis(g)]
    rank = ne(g) - nv(g) + length(connected_components(g))
    @test length(basis_cycles) == rank
    lengths = count_ones.(basis_cycles)
    @test minimum(lengths) == 5
    @test all(len -> len >= 5, lengths)
    @test gf2_rank(basis_cycles, length(eds)) == rank
    masks = Set{eltype(basis_cycles)}()
    for cyc in simplecycles(DiGraph(g))
        length(cyc) < 3 && continue
        push!(masks, cycle_mask(cyc, edge_index, eltype(basis_cycles)))
    end
    basis_vecs, pivots = gf2_basis(basis_cycles, length(eds))
    for mask in masks
        @test iszero(reduce_with_basis(mask, basis_vecs, pivots))
    end
end

function cycle_uai(tensors::Vector{Matrix{T}}) where {T}
    n = length(tensors)
    d1, d2 = size(tensors[1])
    d1 == d2 || throw(ArgumentError("tensors must be square"))
    cards = fill(d1, n)
    factors = Vector{TensorInference.Factor{T, 2}}(undef, n)
    for i in 1:n
        j = i == n ? 1 : i + 1
        size(tensors[i], 1) == d1 || throw(ArgumentError("dimension mismatch"))
        size(tensors[i], 2) == d1 || throw(ArgumentError("dimension mismatch"))
        factors[i] = TensorInference.Factor((i, j), tensors[i])
    end
    return TensorInference.UAIModel(n, cards, factors)
end

function disjoint_cycle_uai(tensors1::Vector{Matrix{T}}, tensors2::Vector{Matrix{T}}) where {T}
    n1 = length(tensors1)
    n2 = length(tensors2)
    d1 = size(tensors1[1], 1)
    d2 = size(tensors2[1], 1)
    size(tensors1[1], 1) == size(tensors1[1], 2) || throw(ArgumentError("tensors1 must be square"))
    size(tensors2[1], 1) == size(tensors2[1], 2) || throw(ArgumentError("tensors2 must be square"))
    all(t -> size(t, 1) == d1 && size(t, 2) == d1, tensors1) || throw(ArgumentError("dimension mismatch in tensors1"))
    all(t -> size(t, 1) == d2 && size(t, 2) == d2, tensors2) || throw(ArgumentError("dimension mismatch in tensors2"))
    cards = vcat(fill(d1, n1), fill(d2, n2))
    factors = Vector{TensorInference.Factor{T, 2}}(undef, n1 + n2)
    for i in 1:n1
        j = i == n1 ? 1 : i + 1
        factors[i] = TensorInference.Factor((i, j), tensors1[i])
    end
    offset = n1
    for i in 1:n2
        j = i == n2 ? 1 : i + 1
        factors[offset + i] = TensorInference.Factor((offset + i, offset + j), tensors2[i])
    end
    return TensorInference.UAIModel(n1 + n2, cards, factors)
end

edge_key(u::Int, v::Int) = u < v ? (u, v) : (v, u)

function edge_list(g::SimpleGraph)
    eds = Tuple{Int, Int}[]
    for e in edges(g)
        u, v = edge_key(src(e), dst(e))
        push!(eds, (u, v))
    end
    sort!(eds)
    return eds
end

function graph_uai(g::SimpleGraph, bond_dim::Int; rng::AbstractRNG = Random.default_rng())
    eds = edge_list(g)
    edge_index = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(eds)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    end
    factors = TensorInference.Factor{Float64}[]
    for v in vertices(g)
        neis = sort!(collect(neighbors(g, v)))
        vars = [edge_index[(v, u)] for u in neis]
        tensor = rand(rng, ntuple(_ -> bond_dim, length(vars))...)
        push!(factors, TensorInference.Factor((vars...,), tensor))
    end
    return TensorInference.UAIModel(length(eds), fill(bond_dim, length(eds)), factors)
end

exact_weight(uai) = probability(TensorNetworkModel(uai))[]

function run_bp(uai; max_iter::Int = 500, tol::Real = 1e-8)
    bp = BeliefPropgation(uai)
    state, info = belief_propagate(bp; max_iter, tol)
    return bp, state, info, bp_vacuum_weight(bp, state)
end

function random_cyclic_graph(n::Int, m::Int; rng::AbstractRNG = Random.default_rng(), max_tries::Int = 100)
    max_edges = n * (n - 1) ÷ 2
    m > max_edges && throw(ArgumentError("m must be <= $max_edges"))
    for _ in 1:max_tries
        g = SimpleGraph(n)
        pairs = [(u, v) for u in 1:n-1 for v in u+1:n]
        shuffle!(rng, pairs)
        for i in 1:m
            u, v = pairs[i]
            add_edge!(g, u, v)
        end
        if length(connected_components(g)) == 1 && ne(g) >= nv(g)
            return g
        end
    end
    error("failed to sample a connected cyclic graph after $max_tries attempts")
end

@testset "loop expansion on single cycle" begin
    A = [0.2 0.9; 0.9 0.2]
    tensors = [A for _ in 1:5]
    uai = cycle_uai(tensors)
    bp, state, info, bp_weight = run_bp(uai; max_iter=500, tol=1e-10)
    @test info.converged

    exact = tr(A^5)
    @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)

    strategies = [
        ("basis", loop_basis(bp)),
        ("xor", loop_series(bp, XORLoopSum(1))),
        ("union", loop_series(bp, UnionLoopSum(1))),
        ("degree", loop_series(bp, Degree(5))),
        ("cyclomatic", loop_series(bp, Cyclomatic(1))),
    ]
    for (name, loops) in strategies
        @testset "$name" begin
            @test length(loops) == 1
            @test count_ones(loops[1].edges) == 5
            result = loop_corrections(bp, state; loops)
            @info "exact: $exact, BP: $bp_weight, loop corrected: $(result.value)"
            @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)
            @test result.value ≈ exact atol=1e-6
        end
    end
end

@testset "loop expansion on disjoint cycles" begin
    A = [0.2 0.9; 0.9 0.2]
    tensors1 = [A for _ in 1:5]
    tensors2 = [A for _ in 1:5]
    uai = disjoint_cycle_uai(tensors1, tensors2)
    bp, state, info, bp_weight = run_bp(uai; max_iter=500, tol=1e-10)
    @test info.converged
    exact = tr(A^5)^2
    @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)

    strategies = [
        ("basis", loop_basis(bp)),
        ("degree", loop_series(bp, Degree(5))),
        ("cyclomatic", loop_series(bp, Cyclomatic(1))),
        ("xor", loop_series(bp, XORLoopSum(1))),
        ("union", loop_series(bp, UnionLoopSum(2))),
    ]
    for (name, loops) in strategies
        @testset "$name" begin
            @test length(loops) == 2
            result_single = loop_corrections(bp, state; loops, n_edges_trunc = 5, n_loops_trunc = 1)
            @test !isapprox(result_single.value, exact; atol=1e-6, rtol=1e-6)
            result_multi = loop_corrections(bp, state; loops, n_edges_trunc = 10, n_loops_trunc = 2)
            @test result_multi.value ≈ exact atol=1e-6
        end
    end
end

@testset "loop expansion on Petersen graph" begin
    rng = MersenneTwister(42)
    g = Graphs.SimpleGraphs.smallgraph(:petersen)
    uai = graph_uai(g, 2; rng)
    bp, state, info, bp_weight = run_bp(uai; max_iter=500, tol=1e-8)
    @test info.converged

    exact = exact_weight(uai)
    @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)

    for trunc in [Degree(12), Cyclomatic(4)] 
        @testset "$(nameof(typeof(trunc)))" begin
            @time loops = loop_series(bp, trunc)
            @test !isempty(loops)
            @time result = loop_corrections(bp, state; loops)
            @info "exact: $exact, BP: $bp_weight, loop corrected: $(result.value)"
            @test result.value ≈ exact atol=1e-6
        end
    end
end

@testset "loop expansion on random simple graphs" begin
    for (n, m, seed) in ((6, 7, 23), (7, 9, 41))
        rng = MersenneTwister(seed)
        g = random_cyclic_graph(n, m; rng)
        uai = graph_uai(g, 2; rng)
        bp, state, info, bp_weight = run_bp(uai; max_iter=1600, tol=1e-12)
        @test info.converged
        exact = exact_weight(uai)
        @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)
        for trunc in (Degree(ne(g)), Cyclomatic(1))
            @testset "$(nameof(typeof(trunc)))" begin
                loops = loop_series(bp, trunc)
                @test !isempty(loops)
                result = loop_corrections(bp, state; loops)
                @test isfinite(bp_weight) && isfinite(result.value) && isfinite(exact)
            end
        end
    end
end

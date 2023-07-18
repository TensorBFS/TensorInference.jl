# MMAP : computing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \ Xₘ.

# A cluster is the elimination of a subset of variables.
struct Cluster{LT}
    eliminated_vars::Vector{LT}
    code::AbstractEinsum
    tensors::Vector
end

"""
$(TYPEDEF)

Computing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \\ Xₘ.

```math
{\\rm MMAP}(X_i|E=e) = \\arg \\max_{X_M} \\sum_{X_S} \\prod_{F} f(x_M, x_S, e)
```

### Fields
* `vars` is the query variables in the tensor network.
* `code` is the tropical tensor network contraction pattern.
* `tensors` is the tensors fed into the tensor network.
* `clusters` is the clusters, each element of this cluster is a [`TensorNetworkModel`](@ref) instance for marginalizing certain variables.
* `evidence` is a dictionary to specifiy degree of freedoms fixed to certain values, which should not have overlap with the query variables.
"""
struct MMAPModel{LT, AT <: AbstractArray}
    vars::Vector{LT}
    code::AbstractEinsum
    tensors::Vector{AT}
    clusters::Vector{Cluster{LT}}
    evidence::Dict{LT, Int}
end

function Base.show(io::IO, mmap::MMAPModel)
    open = getiyv(mmap.code)
    variables = join([string_var(var, open, mmap.evidence) for var in mmap.vars], ", ")
    tc, sc, rw = contraction_complexity(mmap)
    println(io, "$(typeof(mmap))")
    println(io, "variables: $variables")
    println(io, "query variables: $(map(x->x.eliminated_vars, mmap.clusters))")
    print_tcscrw(io, tc, sc, rw)
end
Base.show(io::IO, ::MIME"text/plain", mmap::MMAPModel) = Base.show(io, mmap)

"""
$(TYPEDSIGNATURES)
"""
get_vars(mmap::MMAPModel) = mmap.vars

"""
$(TYPEDSIGNATURES)
"""
function get_cards(mmap::MMAPModel; fixedisone = false)
    vars = get_vars(mmap)
    [fixedisone && haskey(mmap.evidence, vars[k]) ? 1 : length(mmap.tensors[k]) for k in 1:length(vars)]
end

"""
$(TYPEDSIGNATURES)
"""
function MMAPModel(instance::UAIInstance; openvars = (), optimizer = GreedyMethod(), queryvars, evidence=Dict{Int,Int}(), simplifier = nothing)::MMAPModel
    return MMAPModel(
        1:(instance.nvars), instance.cards, instance.factors; queryvars, evidence, optimizer, simplifier, openvars
    )
end

"""
$(TYPEDSIGNATURES)
"""
function MMAPModel(vars::AbstractVector{LT}, cards::AbstractVector{Int}, factors::Vector{<:Factor{T}}; queryvars, openvars = (),
            evidence = Dict{LT, Int}(),
            optimizer = GreedyMethod(), simplifier = nothing,
            marginalize_optimizer = GreedyMethod(), marginalize_simplifier = nothing
        )::MMAPModel where {T, LT}
    all_ixs = [[[var] for var in vars]..., [[factor.vars...] for factor in factors]...]  # labels for vertex tensors (unity tensors) and edge tensors
    iy = collect(LT, openvars)
    evidencevars = collect(LT, keys(evidence))
    marginalized = setdiff(vars, iy ∪ queryvars ∪ evidencevars)
    if !isempty(setdiff(iy, marginalized))
        error("Marginalized variables should not contain any output variable, got $(marginalized) and $iy.")
    end
    all_tensors = [[ones(T, cards[i]) for i in 1:length(vars)]..., getfield.(factors, :vals)...]
    size_dict = OMEinsum.get_size_dict(all_ixs, all_tensors)

    # detect clusters for marginalize variables
    subsets = connected_clusters(all_ixs, marginalized)
    clusters = Cluster{LT}[]
    ixs = Vector{LT}[]
    for (contracted, cluster) in subsets
        ts = all_tensors[cluster]
        ixsi = all_ixs[cluster]
        vari = unique!(vcat(ixsi...))
        iyi = setdiff(vari, contracted)
        codei = optimize_code(EinCode(ixsi, iyi), size_dict, marginalize_optimizer, marginalize_simplifier)
        push!(ixs, iyi)
        push!(clusters, Cluster(contracted, codei, ts))
    end
    rem_indices = setdiff(1:length(all_ixs), vcat([c.second for c in subsets]...))
    remaining_tensors = all_tensors[rem_indices]
    code = optimize_code(EinCode([all_ixs[rem_indices]..., ixs...], iy), size_dict, optimizer, simplifier)
    return MMAPModel(setdiff(vars, marginalized), code, remaining_tensors, clusters, evidence)
end

function OMEinsum.contraction_complexity(mmap::MMAPModel{LT}) where {LT}
    # extract size
    size_dict = Dict(zip(get_vars(mmap), get_cards(mmap; fixedisone = true)))
    sc = -Inf
    tcs = Float64[]
    rws = Float64[]
    for cluster in mmap.clusters
        # update variable sizes
        for k in 1:length(cluster.eliminated_vars)
            # the head sector are for unity tensors.
            size_dict[cluster.eliminated_vars[k]] = length(cluster.tensors[k])
        end
        tc, sci, rw = contraction_complexity(cluster.code, size_dict)
        push!(tcs, tc)
        push!(rws, rw)
        sc = max(sc, sci)
    end

    tc, sci, rw = contraction_complexity(mmap.code, size_dict)
    push!(tcs, tc)
    push!(rws, tc)
    OMEinsum.OMEinsumContractionOrders.log2sumexp2(tcs), max(sc, sci), OMEinsum.OMEinsumContractionOrders.log2sumexp2(rws)
end

function adapt_tensors(mmap::MMAPModel; usecuda, rescale)
    return [adapt_tensors(mmap.code, mmap.tensors, mmap.evidence; usecuda, rescale)...,
        map(cluster -> probability(cluster; evidence = mmap.evidence, usecuda, rescale), mmap.clusters)...]
end

# find connected clusters
function connected_clusters(ixs, vars::Vector{LT}) where {LT}
    visited_ixs = falses(length(ixs))
    visited_vars = falses(length(vars))
    clusters = Pair{Vector{LT}, Vector{Int}}[]
    for (kv, var) in enumerate(vars)
        visited_vars[kv] && continue
        visited_vars[kv] = true
        cluster = LT[] => Int[]
        visit_var!(var, vars, ixs, visited_ixs, visited_vars, cluster)
        sort!(cluster.second)  # sort the tensor indices
        if !isempty(cluster)
            push!(clusters, cluster)
        end
    end
    return clusters
end

function visit_var!(var, vars::AbstractVector{LT}, ixs, visited_ixs, visited_vars, cluster::Pair) where {LT}
    push!(cluster.first, var)
    # add all tensors have not been visited and contain var.
    included_vars = LT[]
    for (kx, ix) in enumerate(ixs)
        if !visited_ixs[kx] && var ∈ ix
            visited_ixs[kx] = true
            push!(cluster.second, kx)

            # new included vars
            for l in ix
                if l != var && l ∈ vars
                    kv = findfirst(==(l), vars)
                    if !visited_vars[kv]
                        visited_vars[kv] = true
                        push!(included_vars, l)
                    end
                end
            end
        end
    end
    # recurse over new variables
    for nvar in included_vars
        visit_var!(nvar, vars, ixs, visited_ixs, visited_vars, cluster)
    end
end

function most_probable_config(mmap::MMAPModel; usecuda = false)::Tuple{Real, Vector}
    vars = get_vars(mmap)
    tensors = map(t -> OMEinsum.asarray(Tropical.(log.(t)), t), adapt_tensors(mmap; usecuda, rescale = false))
    logp, grads = cost_and_gradient(mmap.code, tensors)
    # use Array to convert CuArray to CPU arrays
    return content(Array(logp)[]), map(k -> haskey(mmap.evidence, vars[k]) ? mmap.evidence[vars[k]] : argmax(grads[k]) - 1, 1:length(vars))
end

function maximum_logp(mmap::MMAPModel; usecuda = false)::AbstractArray{<:Real}
    tensors = map(t -> OMEinsum.asarray(Tropical.(log.(t)), t), adapt_tensors(mmap; usecuda, rescale = false))
    return broadcasted_content(mmap.code(tensors...))
end

function log_probability(mmap::MMAPModel, config::Union{Dict, AbstractVector}; rescale = true, usecuda = false)::Real
    @assert length(get_vars(mmap)) == length(config)
    evidence = config isa AbstractVector ? Dict(zip(get_vars(mmap), config)) : config
    assign = merge(mmap.evidence, evidence)
    # two contributions to the probability, not-clustered tensors and clusters.
    m1 = sum(x -> log(x[2][(getindex.(Ref(assign), x[1]) .+ 1)...]), zip(getixsv(mmap.code), mmap.tensors))
    m2 = sum(mmap.clusters) do cluster
        p = probability(cluster; evidence, usecuda, rescale)
        rescale ? p.log_factor : log(p[])
    end
    return m1 + m2
end

function probability(c::Cluster; evidence, usecuda, rescale)::AbstractArray
    tensors = adapt_tensors(c.code, c.tensors, evidence; usecuda, rescale)
    return c.code(tensors...)
end

function log_probability(c::Cluster, config::Union{AbstractVector, Dict})::AbstractArray
    assign = config isa AbstractVector ? Dict(zip(get_vars(c), config)) : config
    return sum(x -> log(x[2][(getindex.(Ref(assign), x[1]) .+ 1)...]), zip(getixsv(c.code), c.tensors))
end

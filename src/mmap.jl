# MMAP : computing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \ Xₘ.

struct MMAPModeling{LT}
    vars::Vector{LT}
    code::AbstractEinsum
    submodels::Vector{TensorNetworkModeling}
    fixedvertices::Dict{LT,Int}
end

"""
$(TYPEDSIGNATURES)
"""
function MMAPModeling(vars::AbstractVector{LT}, factors::Vector{<:Factor{T}}; marginalizedvertices, openvertices=(),
            fixedvertices=Dict{LT,Int}(),
            optimizer=GreedyMethod(), simplifier=nothing,
            marginalize_optimizer=GreedyMethod(), marginalize_simplifier=nothing
        )::MMAPModeling where {T,LT}
    all_ixs = [[[var] for var in vars]..., [[factor.vars...] for factor in factors]...]  # labels for vertex tensors (unity tensors) and edge tensors
    iy = collect(LT, openvertices)
    if !isempty(setdiff(iy, vars))
        error("Marginalized variables should not contain any output variable.")
    end
    all_tensors = [[ones(T, 2) for _=1:length(vars)]..., getfield.(factors, :vals)...]
    size_dict = OMEinsum.get_size_dict(all_ixs, all_tensors)

    # detect clusters for marginalize variables
    clusters = connected_clusters(all_ixs, iy, get_vars(tn))
    models = TensorNetworkModeling[]
    ixs = Vector{LT}[]
    for (contracted, cluster) in clusters
        ts = all_tensors[cluster]
        ixsi = tn.tensors[cluster]
        vari = unique!(vcat(ixsi...))
        iyi = setdiff(vari, contracted)
        codei = optimize_code(EinCode(ixsi, iyi), size_dict, marginalize_optimizer, marginalize_simplifier)
        push!(models, TensorNetworkModeling(vari, codei, ts, fixedvertices))
    end
    code = optimize_code(EinCode(ixs, iy), size_dict, optimizer, simplifier)
    return MMAPModeling(setdiff(get_vars(tn), vars), code, models, fixedvertices)
end

function maximum_logp(mmap::MMAPModeling)
    tensors = map(model->Tropical.(log.(probability(model))), submodels)
    return maximum_logp(TensorNetworkModeling(mmap.vars, mmap.code, tensors, mmap.fixedvertices))
end

function connected_clusters(ixs, vars::Vector{LT}) where LT
    visited_ixs = falses(length(ixs))
    visited_vars = falses(length(vars))
    clusters = Pair{Vector{LT}, Vector{Int}}[]
    for (kv, var) in enumerate(vars)
        visited_vars[kv] && continue
        visited_vars[kv] = true
        cluster = LT[]=>Int[]
        visit_var!(var, vars, ixs, visited_ixs, visited_vars, cluster)
        if !isempty(cluster)
            push!(clusters, cluster)
        end
    end
    return clusters
end

function visit_var!(var, vars::AbstractVector{LT}, ixs, visited_ixs, visited_vars, cluster::Pair) where LT
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

function count_vars(ixs::Vector{Vector{LT}}) where LT
    total_counts = Dict{LT,Int}()
    for ix in [ixs..., iy]
        for label in ix
            total_counts[label] = get(total_counts, label, 0) + 1
        end
    end
    return total_counts
end

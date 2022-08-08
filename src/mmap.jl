# MMAP : computing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \ Xₘ.

"""
$(TYPEDEF)

Computing the most likely assignment to the query variables,  Xₘ ⊆ X after marginalizing out the remaining variables Xₛ = X \\ Xₘ.

```math
{\\rm MMAP}(X_i|E=e) = \\arg \\max_{X_M} \\sum_{X_S} \\prod_{F} f(x_M, x_S, e)
```

### Fields
* `vars` is the remaining (or not marginalized) degree of freedoms in the tensor network.
* `code` is the tropical tensor network contraction pattern.
* `tensors` is the tensors fed into the tensor network.
* `models` is the clusters, each element of this cluster is a [`TensorNetworkModeling`](@ref) instance for marginalizing certain variables.
* `fixedvertices` is a dictionary to specifiy degree of freedoms fixed to certain values, which should not have overlap with the marginalized variables.
"""
struct MMAPModeling{LT,AT<:AbstractArray}
    vars::Vector{LT}
    code::AbstractEinsum
    tensors::Vector{AT}
    models::Vector{TensorNetworkModeling}
    fixedvertices::Dict{LT,Int}
end

"""
$(TYPEDSIGNATURES)
"""
get_vars(mmap::MMAPModeling) = mmap.vars

"""
$(TYPEDSIGNATURES)
"""
function MMAPModeling(instance::UAIInstance; marginalizedvertices, openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)::MMAPModeling
    return MMAPModeling(1:instance.nvars, instance.factors; marginalizedvertices, fixedvertices=Dict(zip(instance.obsvars, instance.obsvals .- 1)), optimizer, simplifier, openvertices)
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
    clusters = connected_clusters(all_ixs, marginalizedvertices)
    models = TensorNetworkModeling[]
    ixs = Vector{LT}[]
    for (contracted, cluster) in clusters
        ts = all_tensors[cluster]
        ixsi = all_ixs[cluster]
        vari = unique!(vcat(ixsi...))
        iyi = setdiff(vari, contracted)
        codei = optimize_code(EinCode(ixsi, iyi), size_dict, marginalize_optimizer, marginalize_simplifier)
        push!(ixs, iyi)
        push!(models, TensorNetworkModeling(vari, codei, ts, fixedvertices))
    end
    rem_indices = setdiff(1:length(all_ixs), vcat([c.second for c in clusters]...))
    remaining_tensors = all_tensors[rem_indices]
    code = optimize_code(EinCode([all_ixs[rem_indices]..., ixs...], iy), size_dict, optimizer, simplifier)
    return MMAPModeling(setdiff(vars, marginalizedvertices), code, remaining_tensors, models, fixedvertices)
end

generate_tensors(mmap::MMAPModeling; usecuda) = [generate_tensors(mmap.code, mmap.tensors, mmap.fixedvertices; usecuda)..., map(model->probability(model; usecuda), mmap.models)...]

# find connected clusters
function connected_clusters(ixs, vars::Vector{LT}) where LT
    visited_ixs = falses(length(ixs))
    visited_vars = falses(length(vars))
    clusters = Pair{Vector{LT}, Vector{Int}}[]
    for (kv, var) in enumerate(vars)
        visited_vars[kv] && continue
        visited_vars[kv] = true
        cluster = LT[]=>Int[]
        visit_var!(var, vars, ixs, visited_ixs, visited_vars, cluster)
        sort!(cluster.second)  # sort the tensor indices
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

"""
$(TYPEDSIGNATURES)
"""
function most_probable_config(mmap::MMAPModeling; usecuda=false)::Tuple{Tropical,Vector}
    vars = get_vars(mmap)
    tensors = map(t->OMEinsum.asarray(Tropical.(log.(t)), t), generate_tensors(mmap; usecuda))
    logp, grads = cost_and_gradient(mmap.code, tensors)
    return logp[], map(k->haskey(mmap.fixedvertices, vars[k]) ? mmap.fixedvertices[vars[k]] : argmax(grads[k]) - 1, 1:length(vars))
end

"""
$(TYPEDSIGNATURES)
"""
function maximum_logp(mmap::MMAPModeling; usecuda=false)::AbstractArray{<:Tropical}
    tensors = map(t->OMEinsum.asarray(Tropical.(log.(t)), t), generate_tensors(mmap; usecuda))
    return mmap.code(tensors...)
end

"""
$(TYPEDSIGNATURES)
"""
function probability(mmap::MMAPModeling, config)::Real
    fixedvertices = merge(mmap.fixedvertices, Dict(zip(get_vars(mmap), config)))
    assign = Dict(zip(get_vars(mmap), config .+ 1))
    m1 = mapreduce(x->x[2][getindex.(Ref(assign), x[1])...], *, zip(getixsv(mmap.code), mmap.tensors))
    m2 = prod(model->probability(chfixedvertices(model, fixedvertices))[], mmap.models)
    return m1 * m2
end
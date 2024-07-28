############ Sampling ############
"""
$TYPEDEF

### Fields
$TYPEDFIELDS

The sampled configurations are stored in `samples`, which is a vector of vector.
`labels` is a vector of variable names for labeling configurations.
The `setmask` is an boolean indicator to denote whether the sampling process of a variable is complete.
"""
struct Samples{L} <: AbstractVector{SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
    samples::Matrix{Int}  # size is nvars × nsample
    labels::Vector{L}
    setmask::BitVector
end
function set_eliminated!(samples::Samples, eliminated_variables)
    for var in eliminated_variables
        loc = findfirst(==(var), samples.labels)
        samples.setmask[loc] && error("varaible `$var` is already eliminated.")
        samples.setmask[loc] = true
    end
    return samples
end
Base.getindex(s::Samples, i::Int) = view(s.samples, :, i)
Base.length(s::Samples) = size(s.samples, 2)
Base.size(s::Samples) = (size(s.samples, 2),)
function Base.show(io::IO, s::Samples)  # display with PrettyTables
    println(io, typeof(s))
    PrettyTables.pretty_table(io, s.samples', header=s.labels)
end
num_samples(samples::Samples) = size(samples.samples, 2)
eliminated_variables(samples::Samples) = samples.labels[samples.setmask]
is_eliminated(samples::Samples{L}, var::L) where L = samples.setmask[findfirst(==(var), samples.labels)]
function idx4labels(totalset::AbstractVector{L}, labels::AbstractVector{L})::Vector{Int} where L
    map(v->findfirst(==(v), totalset), labels)
end
idx4labels(samples::Samples{L}, lb::L) where L = findfirst(==(lb), samples.labels)
function subset(samples::Samples{L}, labels::AbstractVector{L}) where L
    idx = idx4labels(samples.labels, labels)
    @assert all(i->samples.setmask[i], idx)
    return samples.samples[idx, :]
end

function eliminate_dimensions(x::AbstractArray{T, N}, ix::AbstractVector{L}, el::Pair{<:AbstractVector{L}}) where {T, N, L}
    @assert length(ix) == N
    idx = ntuple(N) do i
        if ix[i] ∈ el.first
            k = el.second[findfirst(==(ix[i]), el.first)] + 1
            k:k
        else
            1:size(x, i)
        end
    end
    return asarray(x[idx...], x)
end

function addbatch(samples::Samples, eliminated_variables)
    uniquelabels = unique!(vcat(ixs..., iy))
    labelmap = Dict(zip(uniquelabels, 1:length(uniquelabels)))
    batchdim = length(labelmap) + 1
    newnewixs = [Int[getindex.(Ref(labelmap), ix)..., batchdim] for ix in newixs]
    newnewiy = Int[getindex.(Ref(labelmap), eliminated_variables)..., batchdim]
    newnewxs = [get_slice(x, dimx, samples.samples[ixloc, :]) for (x, dimx, ixloc) in zip(xs, slice_xs_dim, ix_in_sample)]
end

# type unstable
function get_slice(x::AbstractArray{T}, slicedim, configs::AbstractMatrix) where T
    outdim = setdiff(1:ndims(x), slicedim)
    res = similar(x, [size(x, d) for d in outdim]..., size(configs, 2))
    return get_slice!(res, x, outdim, slicedim, configs)
end

function get_slice!(res, x::AbstractArray{T}, outdim, slicedim, configs::AbstractMatrix) where T
    xstrides = strides(x)
    @inbounds for ci in CartesianIndices(res)
        idx = 1
        # the output dimension part
        for (dim, k) in zip(outdim, ci.I)
            idx += (k-1) * xstrides[dim]
        end
        # the sliced part
        batchidx = ci.I[end]
        for (dim, k) in zip(slicedim, view(configs, :, batchidx))
            idx += k * xstrides[dim]
        end
        res[ci] = x[idx]
    end
    return res
end

"""
$(TYPEDSIGNATURES)

Generate samples from a tensor network based probabilistic model.
Returns a vector of vector, each element being a configurations defined on `get_vars(tn)`.

### Arguments
* `tn` is the tensor network model.
* `n` is the number of samples to be returned.
"""
function sample(tn::TensorNetworkModel, n::Int; usecuda = false, queryvars = get_vars(tn))::Samples
    # generate tropical tensors with its elements being log(p).
    xs = adapt_tensors(tn; usecuda, rescale = false)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{Int, Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(tn.code, xs, size_dict)
    # initialize `y̅` as the initial batch of samples.
    iy = getiyv(tn.code)
    setmask = falses(length(queryvars))
    idx = map(l->findfirst(==(l), queryvars), iy ∩ queryvars)
    setmask[idx] .= true
    indices = StatsBase.sample(CartesianIndices(size(cache.content)), _Weights(vec(cache.content)), n)
    configs = zeros(Int, length(queryvars), n)
    for i=1:n
        configs[idx, i] .= indices[i].I .- 1
    end
    samples = Samples(configs, queryvars, setmask)
    # back-propagate
    env = copy(cache.content)
    fill!(env, one(eltype(env)))
    generate_samples!(tn.code, cache, env, samples, samples.labels, size_dict)
    # set evidence variables
    for (k, v) in tn.evidence
        idx = findfirst(==(k), samples.labels)
        samples.samples[idx, :] .= v
    end
    return samples
end
_Weights(x::AbstractVector{<:Real}) = Weights(x)
function _Weights(x::AbstractArray{<:Complex})
    @assert all(e->abs(imag(e)) < 100*eps(abs(e)), x)
    return Weights(real.(x))
end

function generate_samples!(se::SlicedEinsum, cache::CacheTree{T}, env::AbstractArray{T}, samples, pool, size_dict::Dict) where {T}
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return generate_samples!(se.eins, cache, env, samples, pool, size_dict)
end

# pool is a vector of labels that are not eliminated yet.
function generate_samples!(code::NestedEinsum, cache::CacheTree{T}, env::AbstractArray{T}, samples::Samples{L}, pool::Vector{L}, size_dict::Dict{L}) where {T, L}
    if !(OMEinsum.isleaf(code))
        ixs, iy = getixsv(code.eins), getiyv(code.eins)
        for (subcode, child, ix) in zip(code.args, cache.children, ixs)
            # subenv for the current child, use it to sample and update its cache
            siblings = filter(x->x !== child, cache.children)
            siblings_ixs = filter(x->x !== ix, ixs)
            envcode = optimize_code(EinCode([siblings_ixs..., iy], ix), size_dict, GreedyMethod(; nrepeat=1))
            subenv = einsum(envcode, (getfield.(siblings, :content)..., env), size_dict)

            # get samples
            sample_vars = ix ∩ pool
            probabilities = einsum(DynamicEinCode([ix, ix], sample_vars), (child.content, subenv), size_dict)
            update_samples!(samples, sample_vars, probabilities)
            pool = setdiff(pool, sample_vars)

            # eliminate the sampled variables
            setindex!.(Ref(size_dict), 1, sample_vars)
            subsamples = subset(samples, sample_vars)[:, 1]
            udpate_cache_tree!(code, cache, sample_vars=>subsamples)
            subenv = eliminate_dimensions(subenv, ix, sample_vars=>subsamples)

            # recurse
            generate_samples!(subcode, child, subenv, samples, setdiff(pool, sample_vars), size_dict)
        end
    end
end

# probabilities is a tensor of probabilities for each variable in `vars`.
function update_samples!(samples::Samples, vars::AbstractVector{L}, probabilities::AbstractArray{T, N}) where {L, T, N}
    @assert length(vars) == N
    totalset = CartesianIndices(probabilities)
    eliminated_locs = idx4labels(samples.labels, vars)
    config = StatsBase.sample(totalset, _Weights(vec(probabilities)))
    samples.samples[eliminated_locs, 1] .= config.I .- 1
    set_eliminated!(samples, vars)
end

function udpate_cache_tree!(ne::NestedEinsum, cache::CacheTree{T}, el::Pair{<:AbstractVector{L}}) where {T, L}
    OMEinsum.isleaf(ne) && return
    for (subcode, child, ix) in zip(ne.args, cache.children, getixsv(ne.eins))
        child.content = eliminate_dimensions(child.content, ix, el)
        udpate_cache_tree!(subcode, child, el)
    end
end
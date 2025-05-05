############ Sampling ############
"""
$TYPEDEF

### Fields
$TYPEDFIELDS

The sampled configurations are stored in `samples`, which is a vector of vector.
`labels` is a vector of variable names for labeling configurations.
"""
struct Samples{L} <: AbstractVector{SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
    samples::Matrix{Int}  # size is nvars × nsample
    labels::Vector{L}
end
Base.getindex(s::Samples, i::Int) = view(s.samples, :, i)
Base.length(s::Samples) = size(s.samples, 2)
Base.size(s::Samples) = (size(s.samples, 2),)
function Base.show(io::IO, s::Samples)  # display with PrettyTables
    println(io, typeof(s))
    PrettyTables.pretty_table(io, s.samples', header=s.labels)
end
num_samples(samples::Samples) = size(samples.samples, 2)
function idx4labels(totalset::AbstractVector{L}, labels::AbstractVector{L})::Vector{Int} where L
    map(v->findfirst(==(v), totalset), labels)
end
function subset(samples::Samples{L}, labels::AbstractVector{L}) where L
    idx = idx4labels(samples.labels, labels)
    return view(samples.samples, idx, :)
end

function eliminate_dimensions(x::AbstractArray{T, N}, ix::AbstractVector{L}, el::Pair{<:AbstractVector{L}, <:AbstractVector}) where {T, N, L}
    @assert length(ix) == N
    return x[eliminated_selector(size(x), ix, el.first, el.second)...]
end
function eliminated_size(size0, ix, labels)
    @assert length(size0) == length(ix)
    return ntuple(length(ix)) do i
        ix[i] ∈ labels ? 1 : size0[i]
    end
end
function eliminated_selector(size0, ix, labels, config)
    return ntuple(length(ix)) do i
        if ix[i] ∈ labels
            k = config[findfirst(==(ix[i]), labels)] + 1
            k:k
        else
            1:size0[i]
        end
    end
end
function eliminate_dimensions_addbatch!(x::AbstractArray{T, N}, ix::AbstractVector{L}, el::Pair{<:AbstractVector{L}, <:AbstractMatrix}, batch_label::L) where {T, N, L}
    nbatch = size(el.second, 2)
    @assert length(ix) == N
    res = similar(x, (eliminated_size(size(x), ix, el.first)..., nbatch))
    for ibatch in 1:nbatch
        selectdim(res, N+1, ibatch) .= eliminate_dimensions(x, ix, el.first=>view(el.second, :, ibatch))
    end
    push!(ix, batch_label)
    return res
end
function eliminate_dimensions_withbatch(x::AbstractArray{T, N}, ix::AbstractVector{L}, el::Pair{<:AbstractVector{L}, <:AbstractMatrix}) where {T, N, L}
    nbatch = size(el.second, 2)
    @assert length(ix) == N && size(x, N) == nbatch
    res = similar(x, (eliminated_size(size(x), ix, el.first)))
    for ibatch in 1:nbatch
        selectdim(res, N, ibatch) .= eliminate_dimensions(selectdim(x, N, ibatch), ix[1:end-1], el.first=>view(el.second, :, ibatch))
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

### Keyword Arguments
* `usecuda` is a boolean flag to indicate whether to use CUDA for tensor computation.
* `queryvars` is the variables to be sampled, default is `get_vars(tn)`.
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
    idx = map(l->findfirst(==(l), queryvars), iy ∩ queryvars)
    indices = StatsBase.sample(CartesianIndices(size(cache.content)), _Weights(vec(cache.content)), n)
    configs = zeros(Int, length(queryvars), n)
    for i=1:n
        configs[idx, i] .= indices[i].I .- 1
    end
    samples = Samples(configs, queryvars)
    # back-propagate
    env = similar(cache.content, (size(cache.content)..., n))  # batched env
    fill!(env, one(eltype(env)))
    batch_label = _newindex(OMEinsum.uniquelabels(tn.code))
    code = deepcopy(tn.code)
    iy_env = [OMEinsum.getiyv(code)..., batch_label]
    size_dict[batch_label] = n
    generate_samples!(code, cache, iy_env, env, samples, copy(samples.labels), batch_label, size_dict)  # note: `copy` is necessary
    # set evidence variables
    for (k, v) in setdiff(tn.evidence, queryvars)
        idx = findfirst(==(k), queryvars)
        samples.samples[idx, :] .= v
    end
    return samples
end
_newindex(labels::AbstractVector{<:Union{Int, Char}}) = maximum(labels) + 1
_newindex(::AbstractVector{Symbol}) = gensym(:batch)
_Weights(x::AbstractVector{<:Real}) = Weights(x)
function _Weights(x::AbstractArray{<:Complex})
    @assert all(e->abs(imag(e)) < max(100*eps(abs(e)), 1e-8), x) "Complex probability encountered: $x"
    return Weights(real.(x))
end

function generate_samples!(se::SlicedEinsum, cache::CacheTree{T}, iy_env::Vector{Int}, env::AbstractArray{T}, samples::Samples{L}, pool, batch_label::L, size_dict::Dict{L}) where {T, L}
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return generate_samples!(se.eins, cache, iy_env, env, samples, pool, batch_label, size_dict)
end

# pool is a vector of labels that are not eliminated yet.
function generate_samples!(code::DynamicNestedEinsum, cache::CacheTree{T}, iy_env::Vector{L}, env::AbstractArray{T}, samples::Samples{L}, pool::Vector{L}, batch_label::L, size_dict::Dict{L}) where {T, L}
    @assert length(iy_env) == ndims(env)
    if !(OMEinsum.isleaf(code))
        ixs, iy = getixsv(code.eins), getiyv(code.eins)
        for (subcode, child, ix) in zip(code.args, cache.siblings, ixs)
            # subenv for the current child, use it to sample and update its cache
            siblings = filter(x->x !== child, cache.siblings)
            siblings_ixs = filter(x->x !== ix, ixs)
            iy_subenv = batch_label ∈ ix ? ix : [ix..., batch_label]
            envcode = optimize_code(EinCode([siblings_ixs..., iy_env], iy_subenv), size_dict, GreedyMethod(; nrepeat=1))
            subenv = einsum(envcode, (getfield.(siblings, :content)..., env), size_dict)

            # generate samples
            sample_vars = ix ∩ pool
            if !isempty(sample_vars)
                probabilities = einsum(DynamicEinCode([ix, iy_subenv], [sample_vars..., batch_label]), (child.content, subenv), size_dict)
                for ibatch in axes(probabilities, ndims(probabilities))
                    update_samples!(samples.labels, samples[ibatch], sample_vars, selectdim(probabilities, ndims(probabilities), ibatch))
                end
                setdiff!(pool, sample_vars)

                # eliminate the sampled variables
                setindex!.(Ref(size_dict), 1, sample_vars)
                subsamples = subset(samples, sample_vars)
                udpate_cache_tree!(code, cache, sample_vars=>subsamples, batch_label, size_dict)
                subenv = _eliminate!(subenv, ix, sample_vars=>subsamples, batch_label)
            end

            # recurse
            generate_samples!(subcode, child, iy_subenv, subenv, samples, pool, batch_label, size_dict)
        end
    end
end

function _eliminate!(x, ix, el, batch_label)
    if batch_label ∈ ix
        eliminate_dimensions_withbatch(x, ix, el)
    else
        eliminate_dimensions_addbatch!(x, ix, el, batch_label)
    end
end

# probabilities is a tensor of probabilities for each variable in `vars`.
function update_samples!(labels, sample, vars::AbstractVector{L}, probabilities::AbstractArray{T, N}) where {L, T, N}
    @assert length(vars) == N
    totalset = CartesianIndices(probabilities)
    eliminated_locs = idx4labels(labels, vars)
    config = StatsBase.sample(totalset, _Weights(vec(probabilities)))
    sample[eliminated_locs] .= config.I .- 1
end

function udpate_cache_tree!(ne::NestedEinsum, cache::CacheTree{T}, el::Pair{<:AbstractVector{L}}, batch_label::L, size_dict::Dict{L}) where {T, L}
    OMEinsum.isleaf(ne) && return
    updated = false
    for (subcode, child, ix) in zip(ne.args, cache.siblings, getixsv(ne.eins))
        if any(x->x ∈ el.first, ix)
            updated = true
            child.content = _eliminate!(child.content, ix, el, batch_label)
            udpate_cache_tree!(subcode, child, el, batch_label, size_dict)
        end
    end
    updated && (cache.content = einsum(ne.eins, (getfield.(cache.siblings, :content)...,), size_dict))
end
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

"""
$(TYPEDSIGNATURES)

The backward process for sampling configurations.

### Arguments
* `code` is the contraction code in the current step,
* `env` is the environment tensor,
* `samples` is the samples generated for eliminated variables,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_sampling!(code::EinCode, @nospecialize(xs::Tuple), @nospecialize(env), samples::Samples, size_dict)
    ixs, iy = getixsv(code), getiyv(code)
    el = setdiff(vcat(ixs...), iy) ∩ samples.labels

    # get probability
    prob_code = optimize_code(EinCode([ixs..., iy], el), size_dict, GreedyMethod(; nrepeat=1))
    el_prev = eliminated_variables(samples)
    @show el_prev=>subset(samples, el_prev)[:,1]
    xs = [eliminate_dimensions(x, ix, el_prev=>subset(samples, el_prev)[:,1]) for (ix, x) in zip(ixs, xs)]
    probabilities = einsum(prob_code, (xs..., env), size_dict)
    @show el
    @show normalize(real.(vec(probabilities)), 1)

    # sample from the probability tensor
    totalset = CartesianIndices((map(x->size_dict[x], el)...,))
    eliminated_locs = idx4labels(samples.labels, el)
    config = StatsBase.sample(totalset, _Weights(vec(probabilities)))
    @show eliminated_locs, config.I .- 1
    samples.samples[eliminated_locs, 1] .= config.I .- 1

    # eliminate the sampled variables
    set_eliminated!(samples, el)
    setindex!.(Ref(size_dict), 1, el)
    sub = subset(samples, el)[:, 1]
    @show ixs, el=>sub
    xs = [eliminate_dimensions(x, ix, el=>sub) for (ix, x) in zip(ixs, xs)]

    # update environment
    envs = map(1:length(ixs)) do i
        rest = setdiff(1:length(ixs), i)
        code = optimize_code(EinCode([ixs[rest]..., iy], ixs[i]), size_dict, GreedyMethod(; nrepeat=1))
        einsum(code, (xs[rest]..., env), size_dict)
    end
    @show envs
end

function eliminate_dimensions(x::AbstractArray{T, N}, ix::AbstractVector{L}, el::Pair{<:AbstractVector{L}}) where {T, N, L}
    idx = ntuple(N) do i
        if ix[i] ∈ el.first
            k = el.second[findfirst(==(ix[i]), el.first)] + 1
            k:k
        else
            1:size(x, i)
        end
    end
    @show idx
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
    generate_samples!(tn.code, cache, env, samples, size_dict)
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

function generate_samples!(se::SlicedEinsum, cache::CacheTree{T}, env::AbstractArray{T}, samples, size_dict::Dict) where {T}
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return generate_samples!(se.eins, cache, env, samples, size_dict)
end
function generate_samples!(code::NestedEinsum, cache::CacheTree{T}, env::AbstractArray{T}, samples::Samples, size_dict::Dict) where {T}
    @info "@"
    if !(OMEinsum.isleaf(code))
        @info "non-leaf node"
        @show env
        xs = ntuple(i -> cache.children[i].content, length(cache.children))
        envs = backward_sampling!(code.eins, xs, env, samples, size_dict)
        @show envs
        fucks = map(1:length(code.args)) do k
            @info k
            generate_samples!(code.args[k], cache.children[k], envs[k], samples, size_dict)
            return "fuck"
        end
        @info fucks
        return
    else
        @info "leaf node"
        return
    end
end

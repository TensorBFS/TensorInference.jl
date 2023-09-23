# generate tensors based on which vertices are fixed.
adapt_tensors(gp::TensorNetworkModel; usecuda, rescale) = adapt_tensors(gp.code, gp.tensors, gp.evidence; usecuda, rescale)
function adapt_tensors(code, tensors, evidence; usecuda, rescale)
    ixs = getixsv(code)
    # `ix` is the vector of labels (or a degree of freedoms) for a tensor,
    # if a label in `ix` is fixed to a value, do the slicing to the tensor it associates to.
    map(tensors, ixs) do t, ix
        dims = map(ixi -> ixi ∉ keys(evidence) ? Colon() : ((evidence[ixi] + 1):(evidence[ixi] + 1)), ix)
        t2 = t[dims...]
        t3 = usecuda ? CuArray(t2) : t2
        rescale ? rescale_array(t3) : t3
    end
end

# ######### Inference by back propagation ############
# `CacheTree` stores intermediate `NestedEinsum` contraction results.
# It is a tree structure that isomorphic to the contraction tree,
# `content` is the cached intermediate contraction result.
# `siblings` are the siblings of current node.
struct CacheTree{T}
    content::AbstractArray{T}
    siblings::Vector{CacheTree{T}}
end

function cached_einsum(se::SlicedEinsum, @nospecialize(xs), size_dict)
    # slicing is not supported yet.
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching, got nslices = $(length(se.slicing))! Fallback to `NestedEinsum`."
    end
    return cached_einsum(se.eins, xs, size_dict)
end

# recursively contract and cache a tensor network
function cached_einsum(code::NestedEinsum, @nospecialize(xs), size_dict)
    if OMEinsum.isleaf(code)
        # For a leaf node, cache the input tensor
        y = xs[code.tensorindex]
        return CacheTree(y, CacheTree{eltype(y)}[])
    else
        # For a non-leaf node, compute the einsum and cache the contraction result
        caches = [cached_einsum(arg, xs, size_dict) for arg in code.args]
        # `einsum` evaluates the einsum contraction,
        # Its 1st argument is the contraction pattern,
        # Its 2nd one is a tuple of input tensors,
        # Its 3rd argument is the size dictionary (label as the key, size as the value).
        y = einsum(code.eins, ntuple(i -> caches[i].content, length(caches)), size_dict)
        return CacheTree(y, caches)
    end
end

# computed gradient tree by back propagation
function generate_gradient_tree(se::SlicedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where {T}
    if length(se.slicing) != 0
        @warn "Slicing is not supported for generating masked tree! Fallback to `NestedEinsum`."
    end
    return generate_gradient_tree(se.eins, cache, dy, size_dict)
end

# recursively compute the gradients and store it into a tree.
# also known as the back-propagation algorithm.
function generate_gradient_tree(code::NestedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where {T}
    if OMEinsum.isleaf(code)
        return CacheTree(dy, CacheTree{T}[])
    else
        xs = ntuple(i -> cache.siblings[i].content, length(cache.siblings))
        # `einsum_grad` is the back-propagation rule for einsum function.
        # If the forward pass is `y = einsum(EinCode(inputs_labels, output_labels), (A, B, ...), size_dict)`
        # Then the back-propagation pass is
        # ```
        # A̅ = einsum_grad(inputs_labels, (A, B, ...), output_labels, size_dict, y̅, 1)
        # B̅ = einsum_grad(inputs_labels, (A, B, ...), output_labels, size_dict, y̅, 2)
        # ...
        # ```
        # Let `L` be the loss, we will have `y̅ := ∂L/∂y`, `A̅ := ∂L/∂A`...
        dxs = einsum_backward_rule(code.eins, xs, cache.content, size_dict, dy)
        return CacheTree(dy, generate_gradient_tree.(code.args, cache.siblings, dxs, Ref(size_dict)))
    end
end

# a unified interface of the backward rules for real numbers and tropical numbers
function einsum_backward_rule(eins, xs::NTuple{M, AbstractArray{<:Real}} where {M}, y, size_dict, dy)
    return ntuple(i -> OMEinsum.einsum_grad(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), size_dict, dy, i), length(xs))
end

# the main function for generating the gradient tree.
function gradient_tree(code, xs)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(code), xs, Dict{Int, Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(code, xs, size_dict)
    # initialize `y̅` as `1`. Note we always start from `L̅ := 1`.
    dy = match_arraytype(typeof(cache.content), ones(eltype(cache.content), size(cache.content)))
    # back-propagate
    return copy(cache.content), generate_gradient_tree(code, cache, dy, size_dict)
end

# evaluate the cost and the gradient of leaves
function cost_and_gradient(code, xs)
    cost, tree = gradient_tree(code, xs)
    # extract the gradients on leaves (i.e. the input tensors).
    return cost, extract_leaves(code, tree)
end

# since slicing is not supported, we forward it to NestedEinsum.
extract_leaves(code::SlicedEinsum, cache::CacheTree) = extract_leaves(code.eins, cache)

# extract gradients on leaf nodes.
function extract_leaves(code::NestedEinsum, cache::CacheTree)
    res = Vector{Any}(undef, length(getixsv(code)))
    return extract_leaves!(code, cache, res)
end

function extract_leaves!(code, cache, res)
    if OMEinsum.isleaf(code)
        # extract
        res[code.tensorindex] = cache.content
    else
        # resurse deeper
        extract_leaves!.(code.args, cache.siblings, Ref(res))
    end
    return res
end

"""
$(TYPEDSIGNATURES)

Queries the marginals of the variables in a [`TensorNetworkModel`](@ref). The
function returns a dictionary, where the keys are the variables and the values
are their respective marginals. A marginal is a probability distribution over
a subset of variables, obtained by integrating or summing over the remaining
variables in the model. By default, the function returns the marginals of all
individual variables. To specify which marginal variables to query, set the
`mars` field when constructing a [`TensorNetworkModel`](@ref). Note that
the choice of marginal variables will affect the contraction order of the
tensor network.

### Arguments
- `tn`: The [`TensorNetworkModel`](@ref) to query.
- `usecuda`: Specifies whether to use CUDA for tensor contraction.
- `rescale`: Specifies whether to rescale the tensors during contraction.

### Example
The following example is taken from [`examples/asia-network/main.jl`](https://tensorbfs.github.io/TensorInference.jl/dev/generated/asia-network/main/).

```jldoctest; setup = :(using TensorInference, Random; Random.seed!(0))
julia> model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"));

julia> tn = TensorNetworkModel(model; evidence=Dict(1=>0))
TensorNetworkModel{Int64, DynamicNestedEinsum{Int64}, Array{Float64}}
variables: 1 (evidence → 0), 2, 3, 4, 5, 6, 7, 8
contraction time = 2^6.022, space = 2^2.0, read-write = 2^7.077

julia> marginals(tn)
Dict{Vector{Int64}, Vector{Float64}} with 8 entries:
  [8] => [0.450138, 0.549863]
  [3] => [0.5, 0.5]
  [1] => [1.0]
  [5] => [0.45, 0.55]
  [4] => [0.055, 0.945]
  [6] => [0.10225, 0.89775]
  [7] => [0.145092, 0.854908]
  [2] => [0.05, 0.95]

julia> tn2 = TensorNetworkModel(model; evidence=Dict(1=>0), mars=[[2, 3], [3, 4]])
TensorNetworkModel{Int64, DynamicNestedEinsum{Int64}, Array{Float64}}
variables: 1 (evidence → 0), 2, 3, 4, 5, 6, 7, 8
contraction time = 2^7.781, space = 2^5.0, read-write = 2^8.443

julia> marginals(tn2)
Dict{Vector{Int64}, Matrix{Float64}} with 2 entries:
  [2, 3] => [0.025 0.025; 0.475 0.475]
  [3, 4] => [0.05 0.45; 0.005 0.495]
```

In this example, we first set the evidence for variable 1 to 0 and then query
the marginals of all individual variables. The returned dictionary has keys
that correspond to the queried variables and values that represent their
marginals. These marginals are vectors, with each entry corresponding to the
probability of the variable taking a specific value. In this example, the
possible values are 0 or 1. For the evidence variable 1, the marginal is
always [1.0] since its value is fixed at 0.

Next, we specify the marginal variables to query as variables 2 and 3, and
variables 3 and 4, respectively. The joint marginals may or may not affect the
contraction time and space. In this example, the contraction space complexity
increases from 2^{2.0} to 2^{5.0}, and the contraction time complexity
increases from 2^{5.977} to 2^{7.781}. The output marginals are the joint
probabilities of the queried variables, represented by tensors.

"""
function marginals(tn::TensorNetworkModel; usecuda = false, rescale = true)::Dict{Vector{Int}}
    # sometimes, the cost can overflow, then we need to rescale the tensors during contraction.
    cost, grads = cost_and_gradient(tn.code, adapt_tensors(tn; usecuda, rescale))
    @debug "cost = $cost"
    if rescale
        return Dict(zip(tn.mars, LinearAlgebra.normalize!.(getfield.(grads[1:length(tn.mars)], :normalized_value), 1)))
    else
        return Dict(zip(tn.mars, LinearAlgebra.normalize!.(grads[1:length(tn.mars)], 1)))
    end
end

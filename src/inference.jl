# generate tensors based on which vertices are fixed.
function generate_tensors(gp::TensorNetworkModeling)
    fixedvertices = gp.fixedvertices
    isempty(fixedvertices) && return tensors
    ixs = getixsv(gp.code)
    # `ix` is the vector of labels (or a degree of freedoms) for a tensor,
    # if a label in `ix` is fixed to a value, do the slicing to the tensor it associates to.
    map(gp.tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        t[dims...]
    end
end

# ######### Inference by back propagation ############
# `CacheTree` stores intermediate `NestedEinsum` contraction results.
# It is a tree structure that isomorphic to the contraction tree,
# `siblings` are the siblings of current node.
# `content` is the cached intermediate contraction result.
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
        y = einsum(code.eins, ntuple(i->caches[i].content, length(caches)), size_dict)
        return CacheTree(y, caches)
    end
end

# computed gradient tree by back propagation
function generate_gradient_tree(se::SlicedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where T
    if length(se.slicing) != 0
        @warn "Slicing is not supported for generating masked tree! Fallback to `NestedEinsum`."
    end
    return generate_gradient_tree(se.eins, cache, dy, size_dict)
end

# recursively compute the gradients and store it into a tree.
# also known as the back-propagation algorithm.
function generate_gradient_tree(code::NestedEinsum, cache::CacheTree{T}, dy::AbstractArray{T}, size_dict::Dict) where T
    if OMEinsum.isleaf(code)
        return CacheTree(dy, CacheTree{T}[])
    else
        xs = (getfield.(cache.siblings, :content)...,)
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
function einsum_backward_rule(eins, xs::NTuple{M,AbstractArray{<:Real}} where M, y, size_dict, dy)
    return ntuple(i -> OMEinsum.einsum_grad(OMEinsum.getixs(eins), xs, OMEinsum.getiy(eins), size_dict, dy, i), length(xs))
end

# the main function for generating the gradient tree.
function gradient_tree(code, xs)
    # infer size from the contraction code and the input tensors `xs`, returns a label-size dictionary.
    size_dict = OMEinsum.get_size_dict!(getixsv(code), xs, Dict{Int,Int}())
    # forward compute and cache intermediate results.
    cache = cached_einsum(code, xs, size_dict)
    # initialize `y̅` as `1`. Note we always start from `L̅ := 1`.
    dy = fill!(similar(cache.content), one(eltype(cache.content)))
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

Returns the marginal probability distribution of variables.
One can use `get_vars(tn)` to get the full list of variables in this tensor network.
"""
function marginals(tn::TensorNetworkModeling)::Vector
    vars = get_vars(tn)
    _, grads = cost_and_gradient(tn.code, generate_tensors(tn))
    return LinearAlgebra.normalize!.(grads[1:length(vars)], 1)
end
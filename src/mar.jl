# generate tensors based on which vertices are fixed.
adapt_tensors(gp::TensorNetworkModel; usecuda, rescale) = adapt_tensors(gp.code, gp.tensors, gp.evidence; usecuda, rescale)
function adapt_tensors(code, tensors, evidence; usecuda, rescale)
    ixs = getixsv(code)
    # `ix` is the vector of labels (or a degree of freedoms) for a tensor,
    # if a label in `ix` is fixed to a value, do the slicing to the tensor it associates to.
    map(tensors, ixs) do t, ix
        dims = map(ixi -> ixi ∉ keys(evidence) ? Colon() : ((evidence[ixi] + 1):(evidence[ixi] + 1)), ix)
        t2 = t[dims...]
        t3 = usecuda ? togpu(t2) : t2
        rescale ? rescale_array(t3) : t3
    end
end

"""
$(TYPEDSIGNATURES)

Queries the marginals of the variables in a [`TensorNetworkModel`](@ref). The
function returns a dictionary, where the keys are the variables and the values
are their respective marginals. A marginal is a probability distribution over
a subset of variables, obtained by integrating or summing over the remaining
variables in the model. By default, the function returns the marginals of all
individual variables. To specify which marginal variables to query, set the
`unity_tensors_labels` field when constructing a [`TensorNetworkModel`](@ref). Note that
the choice of marginal variables will affect the contraction order of the
tensor network.

### Arguments
- `tn`: The [`TensorNetworkModel`](@ref) to query.

### Keyword Arguments
- `usecuda::Bool`: Specifies whether to use CUDA for tensor contraction.
- `rescale::Bool`: Specifies whether to rescale the tensors during contraction.

### Example
The following example is taken from [`examples/asia-network/main.jl`](https://tensorbfs.github.io/TensorInference.jl/dev/generated/asia-network/main/).

```jldoctest; setup = :(using TensorInference, Random; Random.seed!(0))
julia> model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"));

julia> tn = TensorNetworkModel(model; evidence=Dict(1=>0));

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

julia> tn2 = TensorNetworkModel(model; evidence=Dict(1=>0), unity_tensors_labels = [[2, 3], [3, 4]]);

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
    cost, grads = cost_and_gradient(tn.code, (adapt_tensors(tn; usecuda, rescale)...,))
    @debug "cost = $cost"
    ixs = OMEinsum.getixsv(tn.code)
    queryvars = ixs[tn.unity_tensors_idx]
    if rescale
        return Dict(zip(queryvars, LinearAlgebra.normalize!.(getfield.(grads[tn.unity_tensors_idx], :normalized_value), 1)))
    else
        return Dict(zip(queryvars, LinearAlgebra.normalize!.(grads[tn.unity_tensors_idx], 1)))
    end
end
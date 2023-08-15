using .GenericTensorNetworks: generate_tensors, GraphProblem, flavors, labels

"""
$TYPEDSIGNATURES

Convert a constraint satisfiability problem (or energy model) to a probabilistic model.

### Arguments
* `problem` is a `GraphProblem` instance in [`GenericTensorNetworks`](https://github.com/QuEraComputing/GenericTensorNetworks.jl).
* `β` is the inverse temperature.
"""
function TensorInference.TensorNetworkModel(problem::GraphProblem, β::Real; evidence::Dict=Dict{Int,Int}(),
        optimizer=GreedyMethod(), simplifier=nothing, mars=[[l] for l in labels(problem)])
	ixs = getixsv(problem.code)
	iy = getiyv(problem.code)
    lbs = labels(problem)
	nflavors = length(flavors(problem))
	# generate tensors for x = e^β
	tensors = generate_tensors(exp(β), problem)
    factors = [Factor((ix...,), t) for (ix, t) in zip(ixs, tensors)]
	return TensorNetworkModel(lbs, fill(nflavors, length(lbs)), factors; openvars=iy, evidence, optimizer, simplifier, mars)
end
function TensorInference.MMAPModel(problem::GraphProblem, β::Real;
            queryvars,
            evidence = Dict{labeltype(problem.code), Int}(),
            optimizer = GreedyMethod(), simplifier = nothing,
            marginalize_optimizer = GreedyMethod(), marginalize_simplifier = nothing
        )::MMAPModel
    ixs = getixsv(problem.code)
    iy = getiyv(problem.code)
	nflavors = length(flavors(problem))
	# generate tensors for x = e^β
	tensors = generate_tensors(exp(β), problem)
    factors = [Factor((ix...,), t) for (ix, t) in zip(ixs, tensors)]
    lbs = labels(problem)
    return MMAPModel(lbs, fill(nflavors, length(lbs)), factors; queryvars, openvars=iy, evidence,
        optimizer, simplifier,
        marginalize_optimizer, marginalize_simplifier)
end
 
@info "`TensorInference` loaded `GenericTensorNetworks` extension successfully,
`TensorNetworkModel` and `MMAPModel` can be used for converting a `GraphProblem` to a probabilistic model now."
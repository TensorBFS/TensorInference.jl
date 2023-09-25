module TensorInferenceGTNExt
using TensorInference, TensorInference.OMEinsum
using TensorInference: TYPEDSIGNATURES, Factor
import TensorInference: update_temperature
using GenericTensorNetworks: generate_tensors, GraphProblem, flavors, labels

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

"""
$TYPEDSIGNATURES

Update the temperature of a tensor network model.
The program will regenerate tensors from the problem, without repeated optimizing the contraction order.

### Arguments
- `tnet` is the [`TensorNetworkModel`](@ref) instance.
- `problem` is the target constraint satisfiability problem.
- `β` is the inverse temperature.
"""
function update_temperature(tnet::TensorNetworkModel, problem::GraphProblem, β::Real)
	tensors = generate_tensors(exp(β), problem)
    alltensors = [tnet.tensors[1:end-length(tensors)]..., tensors...]
    return TensorNetworkModel(tnet.vars, tnet.code, alltensors, tnet.evidence, tnet.mars)
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
function update_temperature(tnet::MMAPModel, problem::GraphProblem, β::Real)
    error("We haven't got time to implement setting temperatures for `MMAPModel`.
It is about one or two hours of works. If you need it, please file an issue to let us know: https://github.com/TensorBFS/TensorInference.jl/issues")
end
 
@info "`TensorInference` loaded `GenericTensorNetworks` extension successfully,
`TensorNetworkModel` and `MMAPModel` can be used for converting a `GraphProblem` to a probabilistic model now."
end
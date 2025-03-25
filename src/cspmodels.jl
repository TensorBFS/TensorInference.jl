# generate the tensors for the constraint satisfaction problem
function generate_tensors(β::T, problem::ConstraintSatisfactionProblem) where T <: Real
    cons = ProblemReductions.constraints(problem)
    objs = ProblemReductions.objectives(problem)
    ixs = vcat([t.variables for t in cons], [t.variables for t in objs])
	# generate tensors for x = e^β
    x = exp(β)
    tensors = vcat(
        Array{T}[reshape(map(s -> s ? one(x) : zero(x), t.specification), ntuple(i->num_flavors(problem), length(t.variables))) for t in cons],
        Array{T}[reshape(map(s -> x^s, t.specification), ntuple(i->num_flavors(problem), length(t.variables))) for t in objs]
    )
    return tensors, ixs
end

"""
$TYPEDSIGNATURES

Convert a constraint satisfiability problem (or energy model) to a probabilistic model.

### Arguments
* `problem` is a `ConstraintSatisfactionProblem` instance in [`ProblemReductions`](https://github.com/GiggleLiu/ProblemReductions.jl).
* `β` is the inverse temperature.

### Keyword Arguments
* `evidence` is a dictionary mapping variables to their values.
* `optimizer` is the optimizer used to optimize the tensor network.
* `openvars` is the list of variables to be marginalized.
* `mars` is the list of variables to be marginalized.
"""
function TensorNetworkModel(problem::ConstraintSatisfactionProblem, β::T; evidence::Dict=Dict{Int,Int}(),
        optimizer=GreedyMethod(), openvars=Int[], simplifier=nothing, mars=[[l] for l in variables(problem)]) where T <: Real
    tensors, ixs = generate_tensors(β, problem)
    factors = [Factor((ix...,), t) for (ix, t) in zip(ixs, tensors)]
	return TensorNetworkModel(variables(problem), fill(num_flavors(problem), num_variables(problem)), factors; openvars, evidence, optimizer, simplifier, mars)
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
function update_temperature(tnet::TensorNetworkModel, problem::ConstraintSatisfactionProblem, β::Real)
	tensors, ixs = generate_tensors(β, problem)
    alltensors = [tnet.tensors[1:length(tnet.mars)]..., tensors...]
    return TensorNetworkModel(tnet.vars, tnet.code, alltensors, tnet.evidence, tnet.mars)
end

function MMAPModel(problem::ConstraintSatisfactionProblem, β::Real;
            queryvars,
            openvars = Int[],
            evidence = Dict{Int, Int}(),
            optimizer = GreedyMethod(), simplifier = nothing,
            marginalize_optimizer = GreedyMethod(), marginalize_simplifier = nothing
        )::MMAPModel
	# generate tensors for x = e^β
	tensors, ixs = generate_tensors(β, problem)
    factors = [Factor((ix...,), t) for (ix, t) in zip(ixs, tensors)]
    return MMAPModel(variables(problem), fill(num_flavors(problem), num_variables(problem)), factors; queryvars, openvars, evidence,
        optimizer, simplifier,
        marginalize_optimizer, marginalize_simplifier)
end
function update_temperature(tnet::MMAPModel, problem::ConstraintSatisfactionProblem, β::Real)
    error("We haven't got time to implement setting temperatures for `MMAPModel`.
It is about one or two hours of works. If you need it, please file an issue to let us know: https://github.com/TensorBFS/TensorInference.jl/issues")
end
using TensorInference, Artifacts
using JunctionTrees: read_uai_file, read_uai_evid_file, read_uai_mar_file
using OMEinsum: EinCode, timespace_complexity, get_size_dict, getixsv
using OMEinsumContractionOrders: TreeSA
using LinearAlgebra: normalize!
using Test

@testset "TensorInference.jl" begin

  uai_filepath = joinpath(artifact"MAR_prob", "Promedus_14.uai")
  uai_evid_filepath = joinpath(artifact"MAR_prob", "Promedus_14.uai.evid")
  uai_mar_filepath = joinpath(artifact"MAR_sol", "Promedus_14.uai.MAR")

  nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype=Float32)
  obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
  reference_marginals = read_uai_mar_file(uai_mar_filepath)

  # Does not optimize over open vertices
  rawcode = EinCode([[[i] for i in 1:nvars]..., [[factor.vars...] for factor in factors]...], Int[])  # labels for edge tensors
  tensors = [[ones(Float32, 2) for i=1:length(cards)]..., getfield.(factors, :vals)...]
  tn = TensorNetworksSolver(rawcode, tensors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), optimizer=TreeSA(ntrials=1))
  # @info timespace_complexity(tn.code, get_size_dict(getixsv(tn.code), tensors))
  marginals = gradient(tn.code, generate_tensors(tn))[1:length(cards)] .|> normalize!

  # TODO: add test

end

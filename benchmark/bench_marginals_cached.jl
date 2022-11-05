module BenchMarginalsCached

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

problem = read_uai_problem("Promedus_14")
optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
tn = TensorNetworkModel(problem; optimizer)
SUITE["marginals-cached"] = @benchmarkable marginals(tn)

end  # module
BenchMarginalsCached.SUITE

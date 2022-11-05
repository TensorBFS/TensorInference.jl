module BenchMap

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

problem = read_uai_problem("Promedus_14")

optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
tn = TensorNetworkModel(problem; optimizer)
SUITE["map"] = @benchmarkable most_probable_config(tn)

end  # module
BenchMap.SUITE

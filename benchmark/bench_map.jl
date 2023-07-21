module BenchMap

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

problem = problem_from_artifact("uai2014", "MAR" "Promedus", 14)

optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
tn = TensorNetworkModel(read_model(problem); optimizer, evidence=get_evidence(problem))
SUITE["map"] = @benchmarkable most_probable_config(tn)

end  # module
BenchMap.SUITE

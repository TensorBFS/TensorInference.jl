module BenchMap

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
problem = read_instance(model_filepath; evidence_filepath, solution_filepath)

optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
tn = TensorNetworkModel(problem; optimizer)
SUITE["map"] = @benchmarkable most_probable_config(tn)

end  # module
BenchMap.SUITE

module BenchMap

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

model_filepath, evid_filepath, sol_filepath = get_instance_filepaths("Promedus_14", "MAR")
problem = uai_problem_from_file(model_filepath; uai_evid_filepath = evid_filepath, uai_mar_filepath = sol_filepath)

optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
tn = TensorNetworkModel(problem; optimizer)
SUITE["map"] = @benchmarkable most_probable_config(tn)

end  # module
BenchMap.SUITE

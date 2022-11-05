module BenchMar

using BenchmarkTools
using TensorInference
using Artifacts
# using CUDA
# CUDA.allowscalar(false)

const SUITE = BenchmarkGroup()

problem = read_uai_problem("Promedus_14")

optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
tn1 = TensorNetworkModel(problem; optimizer)
SUITE["mar"] = @benchmarkable marginals(tn1)

# optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
# tn2 = TensorNetworkModel(problem; optimizer)
# SUITE["mar-cuda"] = @benchmarkable marginals(tn2; usecuda = true)

end  # module
BenchMar.SUITE

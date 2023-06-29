module BenchMmap

using BenchmarkTools
using TensorInference
using Artifacts

const SUITE = BenchmarkGroup()

problem = read_uai_problem("Promedus_14")
optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)

# Does not marginalize any var
mmap1 = MMAPModel(problem; marginalized = Int[], optimizer)
SUITE["mmap-1"] = @benchmarkable maximum_logp(mmap1)

# Marginalizes all vars
mmap2 = MMAPModel(problem; marginalized = collect(1:(problem.nvars)), optimizer)
SUITE["mmap-2"] = @benchmarkable maximum_logp(mmap2)

# Does not optimize over open vertices
mmap3 = MMAPModel(problem; marginalized = [2, 4, 6], optimizer)
SUITE["mmap-3"] = @benchmarkable most_probable_config(mmap3)

end  # module
BenchMmap.SUITE

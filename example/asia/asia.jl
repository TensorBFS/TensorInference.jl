using TensorInference

problem = uai_problem_from_file(joinpath(@__DIR__, "data/asia.uai"))
tnet = TensorNetworkModel(problem)
marginals(problem)
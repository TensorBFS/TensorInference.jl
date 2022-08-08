using Test
using OMEinsum
using TensorInference

@testset "map" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)
    instance = read_uai_dir(problem_dir, problem_filename)

    # does not optimize over open vertices
    tn = TensorNetworkModeling(instance; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
    @info timespace_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn)
    @test probability(tn, config) ≈ exp(logp.n)
    @test maximum_logp(tn)[] ≈ logp
end 
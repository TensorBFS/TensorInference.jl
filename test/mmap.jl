using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1,2,3], [2,3,4], [4,5,6]]
    @test TensorInference.connected_clusters(ixs, [2,3,6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "mmap" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)
    instance = read_uai_dir(problem_dir, problem_filename)

    optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40)
    tn_ref = TensorNetworkModeling(instance; optimizer)
    # does not marginalize any var
    tn = MMAPModeling(instance; marginalizedvertices=Int[], optimizer)
    @test maximum_logp(tn_ref) ≈ maximum_logp(tn)

    # marginalize all vars
    tn2 = MMAPModeling(instance; marginalizedvertices=collect(1:nvars), optimizer)
    @test probability(tn_ref)[] ≈ exp(maximum_logp(tn2)[].n)

    # does not optimize over open vertices
    tn3 = MMAPModeling(instance; marginalizedvertices=[2,4,6], optimizer)
    logp, config = most_probable_config(tn3)
    @test probability(tn3, config) ≈ exp(logp.n)
end 
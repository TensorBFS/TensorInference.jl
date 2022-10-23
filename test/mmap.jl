using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1,2,3], [2,3,4], [4,5,6]]
    @test TensorInference.connected_clusters(ixs, [2,3,6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "mmap" begin
    ################# Load problem ####################
    instance = read_uai_problem("Promedus_14")

    optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40)
    tn_ref = TensorNetworkModeling(instance; optimizer)
    # does not marginalize any var
    mmap = MMAPModeling(instance; marginalizedvertices=Int[], optimizer)
    @info(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # marginalize all vars
    mmap2 = MMAPModeling(instance; marginalizedvertices=collect(1:instance.nvars), optimizer)
    @info(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[].n)

    # does not optimize over open vertices
    mmap3 = MMAPModeling(instance; marginalizedvertices=[2,4,6], optimizer)
    @info(mmap3)
    logp, config = most_probable_config(mmap3)
    @test log_probability(mmap3, config) ≈ logp.n
end 

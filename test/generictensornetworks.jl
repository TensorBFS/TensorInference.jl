using Test
using GenericTensorNetworks, TensorInference

@testset "marginals" begin
    # compute the probability
    β = 2.0
    g = GenericTensorNetworks.Graphs.smallgraph(:petersen)
    problem = IndependentSet(g)
    model = TensorNetworkModel(problem, β; mars=[[2, 3]])
    mars = marginals(model)[1]
    problem2 = IndependentSet(g; openvertices=[2,3])
    mars2 = TensorInference.normalize!(GenericTensorNetworks.solve(problem2, PartitionFunction(β)), 1)
    @test mars ≈ mars2

    # mmap
    model = MMAPModel(problem, β; queryvars=[1,4])
    logp, config = most_probable_config(model)
    @test config == [0, 0]
end
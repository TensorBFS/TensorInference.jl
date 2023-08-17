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

    # update temperature
    β2 = 3.0
    model = update_temperature(model, problem, β2)
    pa = probability(model)[]
    model2 = TensorNetworkModel(problem, β2)
    pb = probability(model2)[]
    @test pa ≈ pb

    # mmap
    model = MMAPModel(problem, β; queryvars=[1,4])
    logp, config = most_probable_config(model)
    @test config == [0, 0]
end
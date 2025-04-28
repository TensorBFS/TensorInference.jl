using Test
using TensorInference, ProblemReductions.Graphs
using GenericTensorNetworks

@testset "marginals" begin
    # compute the probability
    β = 2.0
    g = GenericTensorNetworks.Graphs.smallgraph(:petersen)
    problem = IndependentSet(g)
    model = TensorNetworkModel(problem, β; unity_tensors_labels = [[2, 3]])
    mars = marginals(model)[[2, 3]]
    problem2 = IndependentSet(g)
    mars2 = TensorInference.normalize!(GenericTensorNetworks.solve(GenericTensorNetwork(problem2; openvertices=[2, 3]), PartitionFunction(β)), 1)
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

    β = 1.0
    problem = SpinGlass(g, -ones(Int, ne(g)), zeros(Int, nv(g)))
    model = TensorNetworkModel(problem, β; unity_tensors_labels = [[2, 3]])
    samples = sample(model, 100)
    @test sum(energy.(Ref(problem), samples))/100 <= -14
end
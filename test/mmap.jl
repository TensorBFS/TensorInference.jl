using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    @test TensorInference.connected_clusters(ixs, [2, 3, 6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient-based tensor network solvers" begin
    model_filepath, evidence_filepath, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(instance; optimizer)
    # does not marginalize any var
    mmap = MMAPModel(instance; marginalized = Int[], optimizer)
    @debug(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # marginalize all vars
    mmap2 = MMAPModel(instance; marginalized = collect(1:(instance.nvars)), optimizer)
    @debug(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[])

    # does not optimize over open vertices
    mmap3 = MMAPModel(instance; marginalized = [2, 4, 6], optimizer)
    @debug(mmap3)
    logp, config = most_probable_config(mmap3)
    @test log_probability(mmap3, config) ≈ logp
end

@testset "sampling" begin
    instance = TensorInference.read_instance_from_string("""MARKOV
8
 2 2 2 2 2 2 2 2
8
 1 0
 2 1 0
 1 2
 2 3 2
 2 4 2
 3 5 3 1
 2 6 5
 3 7 5 4

2
 0.01
 0.99

4
 0.05 0.01
 0.95 0.99

2
 0.5
 0.5

4
 0.1 0.01
 0.9 0.99

4
 0.6 0.3
 0.4 0.7 

8
 1 1 1 0
 0 0 0 1

4
 0.98 0.05
 0.02 0.95

8
 0.9 0.7 0.8 0.1
 0.1 0.3 0.2 0.9
""")
    n = 10000
    tnet = TensorNetworkModel(instance)
    samples = sample(tnet, n)
    mars = getindex.(marginals(tnet), 2)
    mars_sample = [count(s->s[k]==(1), samples) for k=1:8] ./ n
    @test isapprox(mars, mars_sample, atol=0.05)
end

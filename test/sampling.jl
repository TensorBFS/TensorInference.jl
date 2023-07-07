using TensorInference, Test

@testset "sampling" begin
    instance = TensorInference.uai_problem_from_string("""MARKOV
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
    mars_sample = [count(s->s[k]==(1), samples.samples) for k=1:8] ./ n
    @test isapprox(mars, mars_sample, atol=0.05)
end
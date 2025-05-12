using TensorInference, Test, LinearAlgebra
import StatsBase
using OMEinsum, Random

@testset "sampling" begin
    model = TensorInference.read_model_from_string("""MARKOV
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
    tnet = TensorNetworkModel(model)
    @show sample(tnet, 10)
    samples = sample(tnet, n)
    mars = marginals(tnet)
    mars_sample = [count(s->s[k]==(1), samples) for k=1:8] ./ n
    @test isapprox([mars[[i]][2] for i=1:8], mars_sample, atol=0.05)

    # fix the evidence
    tnet = TensorNetworkModel(model, optimizer=TreeSA(), evidence=Dict(7=>1))
    samples = sample(tnet, n)
    mars = marginals(tnet)
    mars_sample = [count(s->s[k]==(0), samples) for k=1:8] ./ n
    @test isapprox([[mars[[i]][1] for i=1:6]..., mars[[8]][1]], [mars_sample[1:6]..., mars_sample[8]], atol=0.05)
end

@testset "sample MPS" begin
    n = 4
    chi = 3
    Random.seed!(140)
    mps = random_matrix_product_state(n, chi)
    num_samples = 10000
    ixs = OMEinsum.getixsv(mps.code)
    samples = map(1:num_samples) do i
        sample(mps, 1; queryvars=collect(1:n)).samples[:,1]
    end
    samples = sample(mps, num_samples; queryvars=collect(1:n))
    indices = map(samples) do sample
        sum(i->sample[i] * 2^(i-1), 1:n) + 1
    end
    distribution = map(1:2^n) do i
        count(j->j==i, indices) / num_samples
    end
    probs = normalize!(real.(vec(DynamicEinCode(OMEinsum.getixsv(mps.code), collect(1:4))(mps.tensors...))), 1)
    negative_loglikelyhood(probs, samples) = -sum(log.(probs[samples]))/length(samples)
    entropy(probs) = -sum(probs .* log.(probs))
    @show negative_loglikelyhood(probs, indices), entropy(probs)
    @test negative_loglikelyhood(probs, indices) ≈ entropy(probs) atol=1e-1
end
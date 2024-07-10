using TensorInference, Test, StatsBase, OMEinsum, LinearAlgebra

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
    samples = TensorInference.sample(tnet, n)
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

@testset "mps" begin
    T = ComplexF64
    A, B, C, D = randn(T, 2, 2, 2), randn(T, 2, 2, 2), randn(T, 2, 2, 2), randn(T, 2, 2, 1)
    rawket = ein"aib,bjc,ckd,dlm->ijkl"
    ket = optimize_code(rawket, uniformsize(rawket, 10), GreedyMethod())
    p = vec(ket(A, B, C, D))
    s1 = StatsBase.sample(0:15, Weights(abs2.(p)), 1000)
    @test length(s1) == 1000
    Z = (raw=ein"aib,bjc,ckd,dlm,αiβ,βjγ,γkδ,δlε->"; optimize_code(raw, uniformsize(raw, 10), GreedyMethod()))
    tn = TensorNetworkModel(uniquelabels(Z), Z, [A, B, C, D, conj(A), conj(B), conj(C), conj(D)], Dict{Char, Int}(), Vector{Char}[])
    TensorInference.sample(tn, 1000)
end
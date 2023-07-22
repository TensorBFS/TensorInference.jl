using Test
using OMEinsum
using TensorInference, CUDA
CUDA.allowscalar(false)

@testset "gradient-based tensor network solvers" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    model, evidence, reference_solution = read_model(problem), read_evidence(problem), read_solution(problem)

    # does not optimize over open vertices
    tn = TensorNetworkModel(model; optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40), evidence)
    @debug contraction_complexity(tn)
    @time marginals2 = marginals(tn; usecuda = true)
    @test all(x -> x isa CuArray, marginals2)
    # for dangling vertices, the output size is 1.
    npass = 0
    for i in 1:(model.nvars)
        npass += (length(marginals2[i]) == 1 && reference_solution[i] == [0.0, 1]) || isapprox(Array(marginals2[i]), reference_solution[i]; atol = 1e-6)
    end
    @test npass == model.nvars
end

@testset "map" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    model, evidence = read_model(problem), read_evidence(problem)

    # does not optimize over open vertices
    tn = TensorNetworkModel(model; optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40), evidence)
    @debug contraction_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn; usecuda = true)
    @test log_probability(tn, config) ≈ logp
    culogp = maximum_logp(tn; usecuda = true)
    @test culogp isa CuArray
    @test Array(culogp)[] ≈ logp
end

@testset "mmap" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    model, evidence = read_model(problem), read_evidence(problem)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(model; optimizer, evidence)
    # does not marginalize any var
    tn = MMAPModel(model; optimizer, queryvars=collect(1:model.nvars), evidence)
    r1, r2 = maximum_logp(tn_ref; usecuda = true), maximum_logp(tn; usecuda = true)
    @test r1 isa CuArray
    @test r2 isa CuArray
    @test r1 ≈ r2

    # marginalize all vars
    tn2 = MMAPModel(model; optimizer, queryvars=Int[], evidence)
    cup = probability(tn_ref; usecuda = true)
    culogp = maximum_logp(tn2; usecuda = true)
    @test cup isa RescaledArray{T, N, <:CuArray} where {T, N}
    @test culogp isa CuArray
    @test Array(cup)[] ≈ exp(Array(culogp)[])

    # does not optimize over open vertices
    tn3 = MMAPModel(model; optimizer, queryvars=setdiff(1:model.nvars, [2, 4, 6]), evidence)
    logp, config = most_probable_config(tn3; usecuda = true)
    @test log_probability(tn3, config) ≈ logp
end

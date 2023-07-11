using Test
using OMEinsum
using TensorInference, CUDA
CUDA.allowscalar(false)

@testset "gradient-based tensor network solvers" begin
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

    # does not optimize over open vertices
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40))
    @debug contraction_complexity(tn)
    @time marginals2 = marginals(tn; usecuda = true)
    @test all(x -> x isa CuArray, marginals2)
    # for dangling vertices, the output size is 1.
    npass = 0
    for i in 1:(instance.nvars)
        npass += (length(marginals2[i]) == 1 && instance.reference_solution[i] == [0.0, 1]) || isapprox(Array(marginals2[i]), instance.reference_solution[i]; atol = 1e-6)
    end
    @test npass == instance.nvars
end

@testset "map" begin
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

    # does not optimize over open vertices
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40))
    @debug contraction_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn; usecuda = true)
    @test log_probability(tn, config) ≈ logp
    culogp = maximum_logp(tn; usecuda = true)
    @test culogp isa CuArray
    @test Array(culogp)[] ≈ logp
end

@testset "mmap" begin
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(instance; optimizer)
    # does not marginalize any var
    tn = MMAPModel(instance; marginalized = Int[], optimizer)
    r1, r2 = maximum_logp(tn_ref; usecuda = true), maximum_logp(tn; usecuda = true)
    @test r1 isa CuArray
    @test r2 isa CuArray
    @test r1 ≈ r2

    # marginalize all vars
    tn2 = MMAPModel(instance; marginalized = collect(1:(instance.nvars)), optimizer)
    cup = probability(tn_ref; usecuda = true)
    culogp = maximum_logp(tn2; usecuda = true)
    @test cup isa RescaledArray{T, N, <:CuArray} where {T, N}
    @test culogp isa CuArray
    @test Array(cup)[] ≈ exp(Array(culogp)[])

    # does not optimize over open vertices
    tn3 = MMAPModel(instance; marginalized = [2, 4, 6], optimizer)
    logp, config = most_probable_config(tn3; usecuda = true)
    @test log_probability(tn3, config) ≈ logp
end

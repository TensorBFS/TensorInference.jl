using Test
using OMEinsum
using TensorInference, CUDA

@testset "gradient based tensor network solvers" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)

    instance = read_uai_dir(problem_dir, problem_filename)

    # does not optimize over open vertices
    tn = TensorNetworkModeling(instance; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
    @info timespace_complexity(tn)
    @time marginals2 = marginals(tn; usecuda=true)
    # for dangling vertices, the output size is 1.
    npass = 0
    for i=1:instance.nvars
        npass += (length(marginals2[i]) == 1 && instance.reference_marginals[i] == [0.0, 1]) || isapprox(marginals2[i], instance.reference_marginals[i]; atol=1e-6)
    end
    @test npass == instance.nvars
end 

@testset "map" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)
    instance = read_uai_dir(problem_dir, problem_filename)

    # does not optimize over open vertices
    tn = TensorNetworkModeling(instance; optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40))
    @info timespace_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn; usecuda=true)
    @show config
    @test probability(tn, config) ≈ exp(logp.n)
    @test maximum_logp(tn; usecuda=true)[] ≈ logp
end 

@testset "mmap" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)
    instance = read_uai_dir(problem_dir, problem_filename)

    optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40)
    tn_ref = TensorNetworkModeling(instance; optimizer)
    # does not marginalize any var
    tn = MMAPModeling(instance; marginalizedvertices=Int[], optimizer)
    @test maximum_logp(tn_ref; usecuda=true) ≈ maximum_logp(tn; usecuda=true)

    # marginalize all vars
    tn2 = MMAPModeling(instance; marginalizedvertices=collect(1:nvars), optimizer)
    @test probability(tn_ref; usecuda=true)[] ≈ exp(maximum_logp(tn2; usecuda=true)[].n)

    # does not optimize over open vertices
    tn3 = MMAPModeling(instance; marginalizedvertices=[2,4,6], optimizer)
    logp, config = most_probable_config(tn3; usecuda=true)
    @test probability(tn3, config) ≈ exp(logp.n)
end 
using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1,2,3], [2,3,4], [4,5,6]]
    @test TensorInference.connected_clusters(ixs, [2,3,6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "mmap" begin
    ################# Load problem ####################
    problem_number = "14"
    problem_filename = joinpath("Promedus_" * problem_number)
    problem_dir = joinpath(pkgdir(TensorInference), "data", problem_number)
    uai_filepath = joinpath(problem_dir, problem_filename * ".uai")
    uai_evid_filepath = joinpath(problem_dir, problem_filename * ".uai.evid")
    uai_mar_filepath = joinpath(problem_dir, problem_filename * ".uai.MAR")
    td_filepath = joinpath(problem_dir, problem_filename * ".td")

    reference_marginals = read_uai_mar_file(uai_mar_filepath)
    obsvars, obsvals = read_uai_evid_file(uai_evid_filepath)
    nvars, cards, nclique, factors = read_uai_file(uai_filepath; factor_eltype=Float64)

    fixedvertices=Dict(zip(obsvars, obsvals .- 1))
    optimizer=TreeSA(ntrials=1, niters=2, βs=1:0.1:40)
    tn_ref = TensorNetworkModeling(1:nvars, factors; fixedvertices, optimizer)
    # does not marginalize any var
    tn = MMAPModeling(1:nvars, factors; marginalizedvertices=Int[], fixedvertices, optimizer)
    @test maximum_logp(tn_ref) ≈ maximum_logp(tn)

    # marginalize all vars
    tn2 = MMAPModeling(1:nvars, factors; marginalizedvertices=collect(1:nvars), fixedvertices, optimizer)
    @test probability(tn_ref)[] ≈ exp(maximum_logp(tn2)[].n)

    # does not optimize over open vertices
    tn3 = MMAPModeling(1:nvars, factors; marginalizedvertices=[2,4,6], fixedvertices, optimizer)
    logp, config = most_probable_config(tn3)
    @test probability(tn3, config) ≈ exp(logp.n)
end 
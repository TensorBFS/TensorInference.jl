using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1,2,3], [2,3,4], [4,5,6]]
    @test TensorInference.connected_clusters(ixs, [2,3,6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient based tensor network solvers" begin
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

    # does not optimize over open vertices
    tn = TensorNetworkModeling(1:nvars, factors; fixedvertices=Dict(zip(obsvars, obsvals .- 1)), optimizer=TreeSA(ntrials=1))
    @info timespace_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn)
    @show config
    @test probability(tn, config) ≈ exp(logp.n)
    @test maximum_logp(tn)[] ≈ logp
end 
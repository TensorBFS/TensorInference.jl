using Test
using OMEinsum
using TensorInference

@testset "gradient-based tensor network solvers" begin
    model = problem_from_artifact("uai2014", "MAR", "Promedus", 14)

    # does not optimize over open vertices
    tn = TensorNetworkModel(read_instance(model);
        evidence=read_evidence(model),
        optimizer = TreeSA(ntrials = 3, niters = 2, βs = 1:0.1:80))
    @debug contraction_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn)
    @test log_probability(tn, config) ≈ logp
    @test maximum_logp(tn)[] ≈ logp
end

@testset "UAI Reference Solution Comparison" begin
    problem = problem_from_artifact("uai2014", "MAP", "Promedas", 70)
    evidence = read_evidence(problem)
    tn = TensorNetworkModel(read_instance(problem); optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100), evidence)
    _, solution = most_probable_config(tn)
    @test solution == read_solution(problem)
end

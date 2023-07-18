using Test
using OMEinsum
using TensorInference

@testset "gradient-based tensor network solvers" begin
    instance = read_instance_from_artifact("uai2014", "Promedus_14", "MAR")

    # does not optimize over open vertices
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 3, niters = 2, βs = 1:0.1:80))
    @debug contraction_complexity(tn)
    most_probable_config(tn)
    @time logp, config = most_probable_config(tn)
    @test log_probability(tn, config) ≈ logp
    @test maximum_logp(tn)[] ≈ logp
end

@testset "UAI Reference Solution Comparison" begin
    problem_name = "Promedas_70"
    @info "Testing: $problem_name"
    instance = read_instance_from_artifact("uai2014", problem_name, "MAP")
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    _, solution = most_probable_config(tn)
    @test solution == instance.reference_solution
end

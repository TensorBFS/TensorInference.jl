using Test
using OMEinsum
using TensorInference

@testset "gradient-based tensor network solvers" begin
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

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
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths(problem_name, "MAP")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    _, solution = most_probable_config(tn)
    @test solution == instance.reference_solution
end

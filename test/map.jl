using Test
using OMEinsum
using TensorInference

@testset "gradient-based tensor network solvers" begin
    model_filepath, evidence_filepath, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
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
    model_filepath, evidence_filepath, solution_filepath = get_instance_filepaths("Promedas_70", "MAP")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)
    ref_sol = read_solution_file(solution_filepath)[2:end]
    tn = TensorNetworkModel(instance; optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    _, ti_sol = most_probable_config(tn)
    @test ti_sol == ref_sol
end

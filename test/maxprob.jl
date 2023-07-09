using Test
using OMEinsum
using TensorInference

@testset "map" begin
    ################# Load problem ####################
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

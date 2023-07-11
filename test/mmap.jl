using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    @test TensorInference.connected_clusters(ixs, [2, 3, 6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient-based tensor network solvers" begin
    model_filepath, evidence_filepath, _, solution_filepath = get_instance_filepaths("Promedus_14", "MAR")
    instance = read_instance(model_filepath; evidence_filepath, solution_filepath)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(instance; optimizer)

    # Does not marginalize any var
    mmap = MMAPModel(instance; queryvars = collect(1:instance.nvars), optimizer)
    @debug(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # Marginalize all vars
    mmap2 = MMAPModel(instance; queryvars = Int[], optimizer)
    @debug(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[])

    # Does not optimize over open vertices
    mmap3 = MMAPModel(instance; queryvars = setdiff(1:instance.nvars, [2, 4, 6]), optimizer)
    @debug(mmap3)
    logp, config = most_probable_config(mmap3)
    @test log_probability(mmap3, config) ≈ logp

end

@testset "UAI Reference Solution Comparison" begin
    problems = [
        ("Segmentation_12", TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)),
        # ("Segmentation_13", TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)), # fails!
        # ("Segmentation_14", TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40))  # fails!
    ]
    for (problem_name, optimizer) in problems
      @info "Testing: $problem_name"
      model_filepath, evidence_filepath, query_filepath, solution_filepath = get_instance_filepaths(problem_name, "MMAP")
      instance = read_instance(model_filepath; evidence_filepath, query_filepath, solution_filepath)
      model = MMAPModel(instance; queryvars = instance.queryvars, optimizer)
      _, solution = most_probable_config(model)
      @test solution == instance.reference_solution
    end
end
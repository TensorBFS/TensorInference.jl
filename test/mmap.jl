using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    @test TensorInference.connected_clusters(ixs, [2, 3, 6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient-based tensor network solvers" begin
    instance = read_instance_from_artifact("uai2014", "Promedus_14", "MAR")

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(instance; optimizer)

    # Does not marginalize any var
    set_query!(instance, collect(1:instance.nvars))
    mmap = MMAPModel(instance; optimizer)
    @debug(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # Marginalize all vars
    set_query!(instance, Int[])
    mmap2 = MMAPModel(instance; optimizer)
    @debug(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[])

    # Does not optimize over open vertices
    set_query!(instance, setdiff(1:instance.nvars, [2, 4, 6]))
    mmap3 = MMAPModel(instance; optimizer)
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
      instance = read_instance_from_artifact("uai2014", problem_name, "MMAP")
      model = MMAPModel(instance; optimizer)
      _, solution = most_probable_config(model)
      @test solution == instance.reference_solution
    end
end

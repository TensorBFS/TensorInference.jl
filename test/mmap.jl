using Test
using OMEinsum
using TensorInference

@testset "clustering" begin
    ixs = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    @test TensorInference.connected_clusters(ixs, [2, 3, 6]) == [[2, 3] => [1, 2], [6] => [3]]
end

@testset "gradient-based tensor network solvers" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    model, evidence = read_model(problem), read_evidence(problem)

    optimizer = TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)
    tn_ref = TensorNetworkModel(model; optimizer, evidence)

    # Does not marginalize any var
    mmap = MMAPModel(model; optimizer, queryvars=collect(1:model.nvars), evidence)
    @debug(mmap)
    @test maximum_logp(tn_ref) ≈ maximum_logp(mmap)

    # Marginalize all vars
    mmap2 = MMAPModel(model; optimizer, queryvars=Int[], evidence)
    @debug(mmap2)
    @test Array(probability(tn_ref))[] ≈ exp(maximum_logp(mmap2)[])

    # Does not optimize over open vertices
    mmap3 = MMAPModel(model; optimizer, queryvars=setdiff(1:model.nvars, [2, 4, 6]), evidence)
    @debug(mmap3)
    logp, config = most_probable_config(mmap3)
    @test log_probability(mmap3, config) ≈ logp
end

@testset "UAI Reference Solution Comparison" begin
    problem_sets = dataset_from_artifact("uai2014")["MMAP"]
    problems = [
        ("Segmentation", 12, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)),
        # ("Segmentation", 13, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40)), # fails!
        # ("Segmentation", 14, TreeSA(ntrials = 1, niters = 2, βs = 1:0.1:40))  # fails!
    ]
    for (problem_set_name, id, optimizer) in problems
        @testset "$(problem_set_name) problem set, id = $id" begin
            problem = problem_sets[problem_set_name][id]
            @info "Testing: $(problem_set_name)_$id"
            model = MMAPModel(read_model(problem); optimizer, evidence=read_evidence(problem), queryvars=read_queryvars(problem))
            _, solution = most_probable_config(model)
            @test solution == read_solution(problem)
        end
    end
end

using Test
using OMEinsum
using KaHyPar
using TensorInference

@testset "composite number" begin
    A = RescaledArray(2.0, [2.0 3.0; 5.0 6.0])
    x = RescaledArray(2.0, [2.0, 3.0])
    op = ein"ij, j -> i"
    @test Array(x) ≈ exp(2.0) .* [2.0, 3.0]
    @test op(Array(A), Array(x)) ≈ Array(op(A, x))
end

@testset "cached, rescaled contract" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    ref_sol = read_solution(problem)
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    evidence = read_evidence(problem)
    tn = TensorNetworkModel(read_instance(problem); optimizer, evidence)
    p1 = probability(tn; usecuda = false, rescale = false)
    p2 = probability(tn; usecuda = false, rescale = true)
    @test p1 ≈ Array(p2)

    # cached contract
    xs = TensorInference.adapt_tensors(tn; usecuda = false, rescale = true)
    size_dict = OMEinsum.get_size_dict!(getixsv(tn.code), xs, Dict{Int, Int}())
    cache = TensorInference.cached_einsum(tn.code, xs, size_dict)
    @test cache.content isa RescaledArray
    @test Array(cache.content) ≈ p1

    # compute marginals
    ti_sol = marginals(tn)
    ref_sol[collect(keys(evidence))] .= fill([1.0], length(evidence)) # imitate dummy vars
    @test isapprox(ti_sol, ref_sol; atol = 1e-5)
end

@testset "UAI Reference Solution Comparison" begin
    problems = dataset_from_artifact("uai2014")["MAR"]
    problem_sets = [
        #("Alchemy", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("CSP", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),
        #("DBN", KaHyParBipartite(sc_target = 25)),
        #("Grids", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        #("linkage", TreeSA(ntrials = 3, niters = 20, βs = 0.1:0.1:40)), # linkage_15 fails
        #("ObjectDetection", TreeSA(ntrials = 1, niters = 5, βs = 1:0.1:100)),
        ("Pedigree", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        #("Promedus", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)), # greedy also works
        #("relational", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
        ("Segmentation", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))  # greedy also works
    ]

    for (problem_set_name, optimizer) in problem_sets
        @testset "$(problem_set_name) problem set" begin
            for (id, problem) in problems[problem_set_name]
                @info "Testing: $(problem_set_name)_$id"
                tn = TensorNetworkModel(read_instance(problem); optimizer, evidence=read_evidence(problem))
                ref_sol = read_solution(problem)
                evidence = read_evidence(problem)

                # does not optimize over open vertices
                sc = contraction_complexity(tn).sc
                sc > 28 && error("space complexity too large! got $(sc)")
                @debug contraction_complexity(tn)
                ti_sol = marginals(tn)
                ref_sol[collect(keys(evidence))] .= fill([1.0], length(evidence)) # imitate dummy vars
                @test isapprox(ti_sol, ref_sol; atol = 1e-4)
            end
        end
    end
end

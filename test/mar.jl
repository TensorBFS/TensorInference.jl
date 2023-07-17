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
    instance = read_instance_from_artifact("uai2014", "Promedus_14", "MAR")
    ref_sol = instance.reference_solution
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    tn = TensorNetworkModel(instance; optimizer)
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
    ref_sol[instance.obsvars] .= fill([1.0], length(instance.obsvars)) # imitate dummy vars
    @test isapprox(ti_sol, ref_sol; atol = 1e-5)
end

@testset "UAI Reference Solution Comparison" begin
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
    for (problem_set, optimizer) in problem_sets
        @testset "$(problem_set) problem set" begin
            # Capture the problem names that belong to the current problem set
            problem_names = TensorInference.get_problem_names("uai2014", problem_set, "MAR")
            for problem_name in problem_names
                @info "Testing: $problem_name"
                @testset "$(problem_name)" begin
                    instance = read_instance_from_artifact("uai2014", problem_name, "MAR")
                    ref_sol = instance.reference_solution
                    obsvars = instance.obsvars

                    # does not optimize over open vertices
                    tn = TensorNetworkModel(instance; optimizer)
                    sc = contraction_complexity(tn).sc
                    sc > 28 && error("space complexity too large! got $(sc)")
                    @debug contraction_complexity(tn)
                    ti_sol = marginals(tn)
                    ref_sol[obsvars] .= fill([1.0], length(obsvars)) # imitate dummy vars
                    @test isapprox(ti_sol, ref_sol; atol = 1e-4)
                end
            end
        end
    end
end

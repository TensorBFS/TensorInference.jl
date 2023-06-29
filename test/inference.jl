using Test, Artifacts
using OMEinsum
using KaHyPar
using TensorInference

@testset "composite number" begin
    A = RescaledArray(2.0, [2.0 3.0; 5.0 6.0])
    x = RescaledArray(2.0, [2.0, 3.0])
    op = ein"ij, j -> i"
    @test Array(x) ≈ exp(2.0) .* [2.0, 3.0]
    @test op(Array(A), Array(x)) ≈ Array(op(A, x))
    println(x)
end

@testset "cached, rescaled contract" begin
    problem = read_uai_problem("Promedus_14")
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    tn = TensorNetworkModel(problem; optimizer)
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
    marginals2 = marginals(tn)
    npass = 0
    for i in 1:(problem.nvars)
        npass += length(marginals2[i]) == 1 || isapprox(marginals2[i], problem.reference_marginals[i]; atol = 1e-6)
    end
    @test npass == problem.nvars
end

@testset "gradient-based tensor network solvers" begin
    @testset "UAI 2014 problem set" begin
        benchmarks = [
            #("Alchemy", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
            #("CSP", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
            #("DBN", KaHyParBipartite(sc_target=25)),
            #("Grids", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), # greedy also works
            #("linkage", TreeSA(ntrials=3, niters=20, βs=0.1:0.1:40)),  # linkage_15 fails
            #("ObjectDetection", TreeSA(ntrials=1, niters=5, βs=1:0.1:100)),  # ObjectDetection_35 fails
            #("Pedigree", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)), # greedy also works
            ("Promedus", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)),  # greedy also works
            #("relational", TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100)),
            ("Segmentation", TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))  # greedy also works
        ]
        #benchmarks = [("relational", fill(1.0, 5), TreeSA(ntrials=1, niters=5, βs=0.1:0.1:100))]
        #benchmarks = [("DBN",fill(1.0, 6), SABipartite(sc_target=25, βs=0.1:0.01:50))]
        for (benchmark, optimizer) in benchmarks
            @testset "$(benchmark) benchmark" begin

                # Capture the problem names that belong to the current benchmark
                rexp = Regex("($(benchmark)_\\d*)(\\.uai)\$")
                problems = readdir(artifact"MAR_prob"; sort = false) |>
                           x -> map(y -> match(rexp, y), x) |> # apply regex
                                x -> filter(!isnothing, x) |> # filter out `nothing` values
                                     x -> map(first, x) # get the first capture of each element

                for problem in problems
                    @show problem

                    @testset "$(problem)" begin
                        problem = read_uai_problem(problem)

                        # does not optimize over open vertices
                        tn = TensorNetworkModel(problem; optimizer)
                        sc = contraction_complexity(tn).sc
                        if sc > 28
                            error("space complexity too large! got $(sc)")
                        end
                        # @info(tn)
                        @info contraction_complexity(tn)
                        marginals2 = marginals(tn)
                        # for dangling vertices, the output size is 1.
                        npass = 0
                        for i in 1:(problem.nvars)
                            npass += length(marginals2[i]) == 1 || isapprox(marginals2[i], problem.reference_marginals[i]; atol = 1e-6)
                        end
                        @test npass == problem.nvars
                    end
                end
            end
        end
    end
end
